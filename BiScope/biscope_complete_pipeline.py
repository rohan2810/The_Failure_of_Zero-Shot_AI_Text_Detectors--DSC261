import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, classification_report, confusion_matrix, roc_curve
)

df_human = pd.read_csv(r"C:\Users\Manorlab\Desktop\Hemanth\261\BiScope\261_data\original_10k_samples.csv")
df_human['label'] = 0
df_human['source'] = 'eurlex_human'
df_human = df_human.rename(columns={'text': 'text'})
print(f"Loaded human data: {len(df_human)} samples")

df_synthetic = pd.read_csv(r"C:\Users\Manorlab\Desktop\Hemanth\261\BiScope\261_data\synthetic_10k_samples.csv")
df_synthetic['label'] = 1
df_synthetic['source'] = df_synthetic['model_name']
df_synthetic = df_synthetic.rename(columns={'synthetic_text': 'text'})
print(f"Loaded synthetic data: {len(df_synthetic)} samples")

df_combined = pd.concat([
    df_human[['celex_id', 'text', 'label', 'source']],
    df_synthetic[['celex_id', 'text', 'label', 'source']]
], ignore_index=True)

df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset:")
print(f"Total: {len(df_combined)} samples")
print(f"Human (0): {(df_combined['label'] == 0).sum()}")
print(f"AI (1): {(df_combined['label'] == 1).sum()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

class BiScopeFeatureExtractor:
    def __init__(self, model_names: List[str]):
        self.device = device
        self.models = []
        self.tokenizers = []

        print("Loading surrogate models")
        for name in model_names:
            print(f"Loading {name}")
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(self.device)
            model.eval()
            self.tokenizers.append(tokenizer)
            self.models.append(model)

    def extract_features(self, text: str, max_length: int = 512) -> Dict:
        all_features = {
            'forward_losses': [],
            'backward_losses': [],
            'forward_stats': [],
            'backward_stats': []
        }

        for model, tokenizer in zip(self.models, self.tokenizers):
            try:
                tokens = tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=max_length).to(self.device)
                input_ids = tokens['input_ids']

                if input_ids.size(1) < 2:
                    all_features['forward_stats'].append(self._compute_statistics([]))
                    all_features['backward_stats'].append(self._compute_statistics([]))
                    continue

                with torch.no_grad():
                    outputs = model(**tokens)
                    logits = outputs.logits

                    forward_losses = []
                    for i in range(input_ids.size(1) - 1):
                        target_token = input_ids[0, i + 1]
                        pred_logits = logits[0, i, :]
                        ce_loss = F.cross_entropy(
                            pred_logits.unsqueeze(0),
                            target_token.unsqueeze(0)
                        )
                        forward_losses.append(ce_loss.item())

                    backward_losses = []
                    for i in range(1, input_ids.size(1)):
                        prev_token = input_ids[0, i - 1]
                        pred_logits = logits[0, i, :]
                        ce_loss = F.cross_entropy(
                            pred_logits.unsqueeze(0),
                            prev_token.unsqueeze(0)
                        )
                        backward_losses.append(ce_loss.item())

                forward_stats = self._compute_statistics(forward_losses)
                backward_stats = self._compute_statistics(backward_losses)

                all_features['forward_losses'].extend(forward_losses)
                all_features['backward_losses'].extend(backward_losses)
                all_features['forward_stats'].append(forward_stats)
                all_features['backward_stats'].append(backward_stats)
            except:
                all_features['forward_stats'].append(self._compute_statistics([]))
                all_features['backward_stats'].append(self._compute_statistics([]))

        return all_features

    def _compute_statistics(self, losses: List[float]) -> Dict:
        if not losses:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        losses_array = np.array(losses)
        return {
            'mean': float(np.mean(losses_array)),
            'std': float(np.std(losses_array)),
            'min': float(np.min(losses_array)),
            'max': float(np.max(losses_array)),
            'median': float(np.median(losses_array))
        }

def extract_feature_vector(features_dict):
    feature_vec = []
    for stats in features_dict['forward_stats']:
        feature_vec.extend([stats['mean'], stats['std'], stats['min'],
                          stats['max'], stats['median']])
    for stats in features_dict['backward_stats']:
        feature_vec.extend([stats['mean'], stats['std'], stats['min'],
                          stats['max'], stats['median']])
    return np.array(feature_vec)

surrogate_models = ["gpt2", "distilgpt2"]
extractor = BiScopeFeatureExtractor(surrogate_models)

print("\n[STEP 3] Extracting Features")
print(f"Processing {len(df_combined)} samples")

X = []
y = []
failed_indices = []

for idx, row in tqdm(df_combined.iterrows(), total=len(df_combined), desc="Extracting"):
    try:
        features = extractor.extract_features(row['text'])
        feature_vec = extract_feature_vector(features)
        X.append(feature_vec)
        y.append(row['label'])
    except Exception as e:
        failed_indices.append(idx)
        if len(failed_indices) < 3:
            print(f"\nError on sample {idx}: {e}")
        continue

X = np.array(X)
y = np.array(y)

print(f"\nFeature extraction complete")
print(f"Successful: {len(X)}")
print(f"Failed: {len(failed_indices)}")
print(f"Feature shape: {X.shape}")

df_combined_clean = df_combined.drop(failed_indices).reset_index(drop=True)

Path("features").mkdir(exist_ok=True)
np.save('features/X_10k.npy', X)
np.save('features/y_10k.npy', y)
df_combined_clean.to_csv('features/metadata_10k.csv', index=False)
print("Features saved")

forward_mean_gpt2 = X[:, 0]
forward_mean_distil = X[:, 5]
backward_mean_gpt2 = X[:, 10]
backward_mean_distil = X[:, 15]

avg_forward_loss = (forward_mean_gpt2 + forward_mean_distil) / 2
avg_backward_loss = (backward_mean_gpt2 + backward_mean_distil) / 2
loss_diff = avg_backward_loss - avg_forward_loss

fpr, tpr, thresholds = roc_curve(y, loss_diff)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Using Forward-Backward Difference")
print(f"Optimal threshold: {optimal_threshold:.4f}")

y_pred_no_train = (loss_diff > optimal_threshold).astype(int)
y_prob_no_train = (loss_diff - loss_diff.min()) / (loss_diff.max() - loss_diff.min())

acc_no_train = accuracy_score(y, y_pred_no_train)
prec_no_train = precision_score(y, y_pred_no_train, zero_division=0)
rec_no_train = recall_score(y, y_pred_no_train, zero_division=0)
f1_no_train = f1_score(y, y_pred_no_train, zero_division=0)
auroc_no_train = roc_auc_score(y, y_prob_no_train)

print(f"\nPerformance Without Training:")
print(f"Accuracy:  {acc_no_train:.4f} ({acc_no_train*100:.2f}%)")
print(f"Precision: {prec_no_train:.4f}")
print(f"Recall:    {rec_no_train:.4f}")
print(f"F1 Score:  {f1_no_train:.4f}")
print(f"AUROC:     {auroc_no_train:.4f}")

print(f"\nClassification Report:")
print(classification_report(y, y_pred_no_train, target_names=['Human', 'AI']))

cm_no_train = confusion_matrix(y, y_pred_no_train)
print(f"\nConfusion Matrix:")
print(f"                Predicted Human  Predicted AI")
print(f"Actual Human         {cm_no_train[0,0]:6d}          {cm_no_train[0,1]:6d}")
print(f"Actual AI            {cm_no_train[1,0]:6d}          {cm_no_train[1,1]:6d}")

df_no_train = df_combined_clean.copy()
df_no_train['predicted_label'] = y_pred_no_train
df_no_train['predicted_prob'] = y_prob_no_train
df_no_train['correct'] = (df_no_train['label'] == df_no_train['predicted_label']).astype(int)
df_no_train['prediction_result'] = df_no_train['correct'].map({1: 'Correct', 0: 'Wrong'})

Path("predictions").mkdir(exist_ok=True)
df_no_train.to_csv('predictions/predictions_no_training.csv', index=False)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

indices = df_combined_clean.index.values
indices_temp, indices_test = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y
)
indices_train, indices_val = train_test_split(
    indices_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} samples (Human={np.sum(y_train==0)}, AI={np.sum(y_train==1)})")
print(f"Valid: {len(X_val)} samples (Human={np.sum(y_val==0)}, AI={np.sum(y_val==1)})")
print(f"Test:  {len(X_test)} samples (Human={np.sum(y_test==0)}, AI={np.sum(y_test==1)})")


clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

clf.fit(X_train, y_train)

y_val_pred = clf.predict(X_val)
y_val_prob = clf.predict_proba(X_val)[:, 1]

val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_auroc = roc_auc_score(y_val, y_val_prob)

print(f"\nValidation Performance:")
print(f"Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"F1 Score:  {val_f1:.4f}")
print(f"AUROC:     {val_auroc:.4f}")

print("\n[STEP 7] Test Set Evaluation")

y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)[:, 1]

acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred)
rec_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
auroc_test = roc_auc_score(y_test, y_test_prob)

print(f"\nTest Set Performance:")
print(f"Accuracy:  {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"Precision: {prec_test:.4f}")
print(f"Recall:    {rec_test:.4f}")
print(f"F1 Score:  {f1_test:.4f}")
print(f"AUROC:     {auroc_test:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Human', 'AI']))

cm_test = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted Human  Predicted AI")
print(f"Actual Human         {cm_test[0,0]:6d}          {cm_test[0,1]:6d}")
print(f"Actual AI            {cm_test[1,0]:6d}          {cm_test[1,1]:6d}")

print("\n[STEP 8] Saving Predictions")

y_all_pred = clf.predict(X)
y_all_prob = clf.predict_proba(X)[:, 1]

df_results = df_combined_clean.copy()
df_results['predicted_label'] = y_all_pred
df_results['predicted_prob'] = y_all_prob
df_results['correct'] = (df_results['label'] == df_results['predicted_label']).astype(int)
df_results['prediction_result'] = df_results['correct'].map({1: 'Correct', 0: 'Wrong'})

df_results['split'] = 'unknown'
df_results.loc[indices_train, 'split'] = 'train'
df_results.loc[indices_val, 'split'] = 'validation'
df_results.loc[indices_test, 'split'] = 'test'

df_results.to_csv('predictions/predictions_with_training_all.csv', index=False)

df_train = df_results[df_results['split'] == 'train']
df_val = df_results[df_results['split'] == 'validation']
df_test = df_results[df_results['split'] == 'test']

df_train.to_csv('predictions/predictions_train.csv', index=False)
df_val.to_csv('predictions/predictions_validation.csv', index=False)
df_test.to_csv('predictions/predictions_test.csv', index=False)

Path("models").mkdir(exist_ok=True)
with open('models/biscope_10k_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

Path("results").mkdir(exist_ok=True)
results = {
    'dataset': {
        'total_samples': len(df_combined_clean),
        'human_samples': int(np.sum(y == 0)),
        'ai_samples': int(np.sum(y == 1)),
        'train_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    },
    'no_training': {
        'accuracy': float(acc_no_train),
        'precision': float(prec_no_train),
        'recall': float(rec_no_train),
        'f1_score': float(f1_no_train),
        'auroc': float(auroc_no_train),
        'confusion_matrix': cm_no_train.tolist()
    },
    'validation': {
        'accuracy': float(val_acc),
        'f1_score': float(val_f1),
        'auroc': float(val_auroc)
    },
    'test': {
        'accuracy': float(acc_test),
        'precision': float(prec_test),
        'recall': float(rec_test),
        'f1_score': float(f1_test),
        'auroc': float(auroc_test),
        'confusion_matrix': cm_test.tolist()
    }
}

with open('results/10k_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nFinal Comparison")

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC'],
    'No Training': [
        f"{acc_no_train:.4f}",
        f"{prec_no_train:.4f}",
        f"{rec_no_train:.4f}",
        f"{f1_no_train:.4f}",
        f"{auroc_no_train:.4f}"
    ],
    'Validation': [
        f"{val_acc:.4f}",
        "-",
        "-",
        f"{val_f1:.4f}",
        f"{val_auroc:.4f}"
    ],
    'Test': [
        f"{acc_test:.4f}",
        f"{prec_test:.4f}",
        f"{rec_test:.4f}",
        f"{f1_test:.4f}",
        f"{auroc_test:.4f}"
    ]
})

print(comparison.to_string(index=False))

improvement = acc_test - acc_no_train
print(f"\nAccuracy improvement from training: +{improvement:.4f} ({improvement*100:.2f} percentage points)")
