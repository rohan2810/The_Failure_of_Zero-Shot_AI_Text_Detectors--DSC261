import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

sns.set_style("whitegrid")

data_file = Path("Dataset/BiScope_Original/train_processed.jsonl")
samples = []
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

df = pd.DataFrame(samples)
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nTask distribution:")
print(df['task'].value_counts())

num_samples = 1000

sampled_df = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), num_samples//2), random_state=42)
)

print(f"\nSampled {len(sampled_df)} samples")
print(f"Human (0): {(sampled_df['label'] == 0).sum()}")
print(f"AI (1): {(sampled_df['label'] == 1).sum()}")

device = "cuda" if torch.cuda.is_available() else "cpu"

class BiScopeFeatureExtractor:
    def __init__(self, model_names: List[str]):
        self.device = device
        self.models = []
        self.tokenizers = []

        for name in model_names:
            print(f"Loading {name}")
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
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

print("\nExtracting BiScope features")

X = []
y = []
failed = 0

for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Extracting"):
    try:
        features = extractor.extract_features(row['text'])
        feature_vec = extract_feature_vector(features)
        X.append(feature_vec)
        y.append(row['label'])
    except Exception as e:
        failed += 1
        if failed < 5:
            print(f"\nError on sample {idx}: {e}")
        continue

X = np.array(X)
y = np.array(y)


Path("features").mkdir(exist_ok=True)
np.save('features/X_train.npy', X)
np.save('features/y_train.npy', y)
print(f"Features saved to features directory")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset split:")
print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

print("\nTraining BiScope classifier")

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

clf.fit(X_train, y_train)
print("\nTraining complete")

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob)

print(f"\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, label=f'BiScope (AUROC={auroc:.4f})',
             linewidth=2.5, color='blue')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
            cbar_kws={'label': 'Count'})
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
axes[2].barh(range(10), importances[indices], color='skyblue')
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([f'Feature {i}' for i in indices])
axes[2].set_xlabel('Importance', fontsize=12)
axes[2].set_title('Top 10 Features', fontsize=14, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('biscope_training_results.png', dpi=300, bbox_inches='tight')
print(f"\nResults saved to biscope_training_results.png")

Path("models").mkdir(exist_ok=True)
with open('models/biscope_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

print(f"\nModel saved to models/biscope_classifier.pkl")

print(f"\nTraining Summary")
print(f"Samples used: {len(X)}")
print(f"Train/Test: {len(X_train)}/{len(X_test)}")
print(f"Features per sample: {X.shape[1]}")
print(f"AUROC: {auroc:.4f}")
print(f"F1: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
