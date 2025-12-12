import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

sns.set_style("whitegrid")
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

surrogate_models = ["gpt2", "distilgpt2"]
print("Initializing BiScope")
extractor = BiScopeFeatureExtractor(surrogate_models)

data_file = Path("Dataset/BiScope_Original/train_processed.jsonl")
samples = []
human_samples = []
ai_samples = []

with open(data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        
        if sample['label'] == 0 and len(human_samples) == 0:
            human_samples.append(sample)
            samples.append(sample)
        
        if sample['label'] == 1 and len(ai_samples) == 0:
            ai_samples.append(sample)
            samples.append(sample)
        
        if len(human_samples) > 0 and len(ai_samples) > 0:
            break

print(f"\nLoaded {len(samples)} samples (Human: {len(human_samples)}, AI: {len(ai_samples)})")

for i, sample in enumerate(samples):
    print(f"\nSample {i+1}:")
    print(f"Task: {sample['task']}")
    print(f"Source: {sample['source']}")
    print(f"Label: {'Human' if sample['label'] == 0 else 'AI'}")
    print(f"Text preview: {sample['text'][:100]}")
    
    features = extractor.extract_features(sample['text'])
    
    print(f"\nFeatures extracted:")
    print(f"Forward losses: {len(features['forward_losses'])} values")
    print(f"Backward losses: {len(features['backward_losses'])} values")
    
    for j, stats in enumerate(features['forward_stats']):
        print(f"Model {j+1} - Forward mean: {stats['mean']:.4f}")
    for j, stats in enumerate(features['backward_stats']):
        print(f"Model {j+1} - Backward mean: {stats['mean']:.4f}")

human_sample = human_samples[0]
ai_sample = ai_samples[0]

print(f"\nHuman sample: {human_sample['task']} - {human_sample['source']}")
print(f"AI sample: {ai_sample['task']} - {ai_sample['source']}")

human_features = extractor.extract_features(human_sample['text'])
ai_features = extractor.extract_features(ai_sample['text'])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

n_tokens = min(200, len(human_features['forward_losses']))
n_tokens_ai = min(200, len(ai_features['forward_losses']))

axes[0, 0].plot(human_features['forward_losses'][:n_tokens], 
                color='blue', alpha=0.7, linewidth=1.5)
axes[0, 0].set_title('Human Text - Forward Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Token Position')
axes[0, 0].set_ylabel('Cross-Entropy Loss')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(human_features['backward_losses'][:n_tokens], 
                color='green', alpha=0.7, linewidth=1.5)
axes[0, 1].set_title('Human Text - Backward Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Token Position')
axes[0, 1].set_ylabel('Cross-Entropy Loss')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(ai_features['forward_losses'][:n_tokens_ai], 
                color='red', alpha=0.7, linewidth=1.5)
axes[1, 0].set_title(f'AI Text ({ai_sample["source"]}) - Forward Loss', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Token Position')
axes[1, 0].set_ylabel('Cross-Entropy Loss')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(ai_features['backward_losses'][:n_tokens_ai], 
                color='orange', alpha=0.7, linewidth=1.5)
axes[1, 1].set_title(f'AI Text ({ai_sample["source"]}) - Backward Loss', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Token Position')
axes[1, 1].set_ylabel('Cross-Entropy Loss')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('biscope_loss_patterns.png', dpi=300, bbox_inches='tight')
print("\nLoss pattern visualization saved to biscope_loss_patterns.png")

print(f"\nFeature Comparison: Human vs AI")

print(f"\nHuman Text:")
for j, stats in enumerate(human_features['forward_stats']):
    print(f"Model {j+1} - Forward mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
for j, stats in enumerate(human_features['backward_stats']):
    print(f"Model {j+1} - Backward mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")

print(f"\nAI Text ({ai_sample['source']}):")
for j, stats in enumerate(ai_features['forward_stats']):
    print(f"Model {j+1} - Forward mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
for j, stats in enumerate(ai_features['backward_stats']):
    print(f"Model {j+1} - Backward mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")

print("\nKey Observation:")
print("Notice the differences in backward loss means")
print("This is what BiScope uses to distinguish human from AI text")
