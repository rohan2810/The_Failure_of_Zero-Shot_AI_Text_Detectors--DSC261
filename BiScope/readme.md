# BiScope:

Implementation of BiScope methodology for detecting AI-generated legal text using bi-directional cross-entropy analysis.


## Key Results

- **Without Training**: 74.5% accuracy using simple threshold-based classification
- **With Training**: 89.2% accuracy using Random Forest classifier
- **Dataset**: 20,000 legal documents (10K human, 10K AI-generated)
- **Models Used**: GPT-2, DistilGPT-2 as surrogate models

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm
```


### Verify Environment

```bash
test_env.py
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
biscope_complete_pipeline.py
```

This script:
1. Loads human and AI-generated legal documents
2. Extracts BiScope features using surrogate models
3. Evaluates without training (threshold-based)
4. Trains Random Forest classifier
5. Generates predictions and evaluation metrics

### Step-by-Step Execution

#### 1. Load and Explore Data
```bash
load_biscope_data.py
```

#### 2. Extract Features
```bash
extract_biscope_features.py
```

This extracts bi-directional cross-entropy features:
- Forward loss: predicting next token
- Backward loss: predicting previous token (BiScope novelty)

#### 3. Train Classifier
```bash
train_biscope_classifier.py
```

Trains Random Forest on extracted features with 70/15/15 train/val/test split.



