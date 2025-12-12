BiScope:

Implementation of BiScope methodology for detecting AI-generated legal text using bi-directional cross-entropy analysis.

Key Results

Without Training: 74.5% accuracy using simple threshold-based classification
With Training: 89.2% accuracy using Random Forest classifier
Dataset: 20,000 legal documents (10K human, 10K AI-generated)
Models Used: GPT-2, DistilGPT-2 as surrogate models

Usage
Quick Start
Run the complete pipeline:
bashpython scripts/05_complete_pipeline.py
This script:

Loads human and AI-generated legal documents
Extracts BiScope features using surrogate models
Evaluates without training (threshold-based)
Trains Random Forest classifier
Generates predictions and evaluation metrics

Step-by-Step Execution
1. Load and Explore Data
bashpython scripts/02_load_data.py
2. Extract Features
bashpython scripts/03_extract_features.py
This extracts bi-directional cross-entropy features:

Forward loss: predicting next token
Backward loss: predicting previous token (BiScope novelty)

3. Train Classifier
bashpython scripts/04_train_classifier.py
Trains Random Forest on extracted features with 70/15/15 train/val/test split.

