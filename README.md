# Credit Card Fraud Detection Capstone

## Project Overview

This capstone project applies machine learning to credit card fraud detection using the Kaggle/ULB Credit Card Fraud dataset. The dataset contains anonymized European cardholder transactions and is highly imbalanced: only about 0.17% of transactions are fraudulent. The primary objective is to build a robust fraud detection pipeline with repeatable engineering, strong validation, and a clear comparison between simple and advanced models.

## Goals

- Demonstrate data engineering and automated preprocessing for fraud detection.
- Compare baseline and advanced models to show real performance gains.
- Maintain reproducibility with fixed random seeds and environment dependencies.
- Provide clear analysis of class imbalance, bias mitigation, and model failure modes.
- Deliver an end-to-end workflow that supports a one-sample inference quick start.

## Dataset

- Source: Kaggle Credit Card Fraud Detection dataset
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Data characteristics:
  - Transactions from European cardholders
  - 284,807 total transactions
  - 492 fraudulent cases (0.17% of transactions)
  - 30 features: `Time`, `V1`–`V28`, `Amount`, and `Class`

## Project Structure

- `data/` - raw and external dataset storage
- `notebooks/` - exploratory data analysis, modeling experiments, and reporting notebooks
- `reports/` - metrics, diagrams, and project documentation outputs
- `src/` - modular Python code for loading data, preprocessing, training, evaluation, and inference
- `requirements.txt` - Python dependencies for reproducible setup

## Project Phases

Phase 1: Project Initiation & Setup – Finalize project proposal, dataset, GitHub setup, and project roadmap.
Phase 2: Data Engineering & EDA – Implement data loading, EDA, and the preprocessing pipeline.
Phase 3: Baseline Regression & Anomaly Detection – Implement a multi-parameter linear regression baseline
* using linear regression as a baseline usually means one of two setups:
    * Target Reconstruction: Using the regression model to predict a continuous feature based on the others, using the prediction error (residual) as an anomaly score.
    * Proxy Classification: Running a standard linear probability model where the targets are 0 and 1 to establish a quick linear decision boundary before jumping into the complex reconstructions of the Autoencoder.
Phase 4: Supervised Modeling & Evaluation – Implement XGBoost and evaluate performance against the baseline.
Phase 5: Validation & Project Closure – Unsupervised Modeling - Autoencoder-based anomaly detection.
Phase 6: Implement GNN
Phase 7: Implement tab transformer
Phase 8: Implement ?
Phase 9: Phase 5: GCP Deployment & Infrastructure – Deploy the solution on Google Cloud Platform (GCP) utilizing Cloud Run and/or Firebase Hosting.
Phase 10: Validate the final model, compile the report, and present.

## Quick Start

### 1. Install environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare data

```powershell
python src/data_loader.py --download --output data/raw
python src/preprocess.py --input data/raw --output data/processed
```

### 3. Train models

```powershell
python src/train.py --config configs/train_config.yaml
```

### 4. Run a single prediction

```powershell
python src/predict.py --model models/final_model.pkl --input examples/sample_transaction.csv
```

> This quick start is designed so a grader can run one sample prediction in under 2 minutes after environment setup.

## Reproducibility

- `requirements.txt` captures exact package versions.
- Random seeds are fixed for NumPy, scikit-learn, and any deep learning frameworks used.
- Training and inference are deterministic where possible.
- The repository is organized into modular code files rather than a single script.

## Data & Bias Analysis

The dataset is severely imbalanced. This project documents:

- class distribution for training and test sets
- feature-level statistics such as mean, variance, and histograms
- oversampling, undersampling, or class-weighted loss strategies used to mitigate imbalance
- explicit measures taken to avoid bias from target leakage or any future information

## Baseline & Model Comparison

Baseline models include:

- Logistic Regression or linear classifier
- simple heuristic / rule-based detector

Advanced models under evaluation:

- XGBoost
- Autoencoder anomaly detection
- Graph Neural Network (GNN) variant for relational risk features
- TabTransformer for tabular feature interactions

Each model is compared using metrics appropriate for imbalanced fraud detection, including:

- Precision
- Recall
- F1 score
- ROC AUC
- PR AUC
- confusion matrix

## Validation Strategy

- k-fold cross-validation is used (typically `k=5`) rather than a single split.
- A separate holdout test set is preserved for final evaluation.
- Data leakage is explicitly prevented by ensuring preprocessing and feature selection happen within cross-validation boundaries.
- Statistical significance is assessed for model comparisons where applicable.

## Performance Metrics & Analysis

This project documents:

- training time for each model
- inference latency per sample
- total trainable parameters for neural architectures
- feature importance and model explainability outputs
- failure mode analysis with specific examples of high-risk misclassifications

## Experiments and Reporting

The project deliverables include:

- `notebooks/` for exploratory analysis and model comparison
- `reports/` with plots and captions for loss curves, precision-recall curves, feature importance, and confusion matrices
- ablation studies for complex model components such as Transformer blocks or GNN layers

## Notes for Graders

- The repository follows a modular structure and is designed for reproducibility.
- All plots include labeled axes, units, and captions.
- This README includes a quick start guide, dataset summary, validation plan, and experimental approach.

## Future Work

Potential next steps include:

- extending the dataset to real-time streaming fraud detection
- integrating additional external risk signals or user metadata
- turning the pipeline into a Dockerized production-ready service
- adding more explainability and compliance reporting for finance/healthcare use cases

## References

- Kaggle Credit Card Fraud dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Fraud detection best practices for imbalanced classification
