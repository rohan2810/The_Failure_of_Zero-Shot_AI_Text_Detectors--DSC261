# Lost in Legal Language: The Failure of Zero-Shot AI Text Detectors

**DSC 261 — Responsible Data Science (UC San Diego)**

**Authors**: Abhilash Shankarampeta, Ananay Gupta, Hemanth Bodala, Rohan Surana, Sarthak Kala, Vignesh Palaniappan

---

## Motivation

Recent legal cases (Mata v. Avianca, Noland v. Land of the Free, Wadsworth v. Walmart) show AI-generated fake citations are causing real professional sanctions. Nearly **100 AI hallucination incidents** have been catalogued across U.S. courts. 

This project evaluates whether modern AI-text detectors (**BiScope**, **Binoculars**, **OOD-DeepSVDD**) can reliably distinguish human-written legal text from LLM-generated content.

---

## Dataset & Synthetic Data

- **Base dataset**: EUR-Lex (10,000 human-written legal documents from `jonathanli/eurlex`)
- **Synthetic generation**: 10,000 documents generated using 4 LLMs (2,500 per model):
  - Qwen3-8B
  - Llama-3.1-8B-Instruct
  - Mistral-7B-Instruct-v0.3
  - Gemma-3-4b-it
- **Strategy**: 50/50 mix of preserving vs. modifying case numbers (shuffled)
- **Generation params**: batch_size=32, temp=0.7, top_p=0.9, max_tokens=2048

---

## Methods

1. **BiScope**: Bi-directional cross-entropy detection using GPT-2/DistilGPT-2 features + Random Forest classifier
2. **Binoculars**: Zero-shot perplexity ratio across two language models
3. **OOD-DeepSVDD**: Out-of-distribution detection using hypersphere learning with RoBERTa embeddings

---

## Results

### Overall Performance (20,000 documents)

| Method | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| BiScope (no train) | 50.1% | 65.5% | 0.6% | 1.1% |
| **BiScope (trained)** | **74.5%** | **79.5%** | **66.0%** | **72.1%** |
| Binoculars | 50.7% | 50.4% | 99.3% | 66.8% |
| OOD-DeepSVDD | 55.8% | 61.0% | 32.1% | 42.1% |

### Generator-Specific Performance (BiScope Recall %)

| Generator | Overall | 0-200w | 200-400w | 400-600w | 600-800w |
|-----------|---------|--------|----------|----------|----------|
| Qwen3-8B | 86.0% | 95.5% | 85.4% | 76.1% | 78.6% |
| Gemma-3-4b | 72.9% | 92.9% | 71.6% | 64.7% | 50.0% |
| Mistral-7B | 55.2% | 67.4% | 58.7% | 43.9% | 27.3% |
| Llama-3.1-8B | 51.1% | 70.3% | 47.9% | 40.9% | 38.1% |
| **Overall** | **66.0%** | **84.2%** | **65.5%** | **53.9%** | **50.6%** |

### Key Findings

- **Generator bias**: Detection varies 51-86% across models (4× difference)
- **Length degradation**: Performance drops from 84.2% (short) to 50.6% (600-800 words)
- **Error rates**: 17% false positives, 34% false negatives
- **Training requirement**: BiScope fails catastrophically without training (0.6% recall)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

Outputs saved to `eurlex_synthetic_data/`:
- `original_10000_samples.csv`
- `synthetic_10000_samples.csv`
---

## Data

Original and generated synthetic data are available in `eurlex_synthetic_data`.

---

## Running Inference

### Binoculars Inference

Run Binoculars detector on the synthetic and original datasets:

```bash
cd Binoculars
python infer.py
```

This processes both datasets and outputs results with detection probabilities to:
- `./outputs/eurlex_synthetic_data/synthetic_10k_samples.csv`
- `./outputs/eurlex_synthetic_data/original_10k_samples.csv`

For more details, see [`Binoculars/README.md`](Binoculars/README.md).

### OOD-DeepSVDD Inference

First, download the pre-trained model weights from [Google Drive](https://drive.google.com/drive/folders/173jObPXmvAS9R0s1PERaSgsbeXlULfHl?usp=sharing).

Then run the OOD detector:

```bash
cd ood-llm-detect
python infer.py --model_path <path_to_model>/model_classifier_best.pth \
                --ood_type deepsvdd \
                --mode deepfake \
                --out_dim 768
```

Results are saved to:
- `./outputs/eurlex_synthetic_data/synthetic_10k_samples.csv`
- `./outputs/eurlex_synthetic_data/original_10k_samples.csv`

For more details, see [`ood-llm-detect/README.md`](ood-llm-detect/README.md).

---