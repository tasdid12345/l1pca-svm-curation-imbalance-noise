# Balanced RBF SVM vs PCA-curated SVM on credit-g (noisy labels)

Motivation: The paper "Training Dataset Curation by L1-Norm Principal-Component Analysis for Support Vector Machines" (2025) proposes curating SVM training sets with L1-norm PCA to remove outlying or noisy points before training. This experiment asks whether a simpler route, class-balanced weighting plus cross-validated hyper-parameters on the full noisy set, does at least as well as an L2-PCA reconstruction-error curation on a small imbalanced dataset.

Question: Does a class-balanced, CV-tuned RBF SVM on the full noisy, imbalanced training set beat an SVM trained on a PCA-curated subset of that set, on the public credit-g dataset?

Dataset: credit-g from OpenML (data_id 31), 1,000 rows, about 70/30 class split. This is a public dataset, not the paper's own. Features are one-hot encoded and standardised; a stratified 75/25 split is fixed by seed 42, and 10% of training labels are flipped at random to simulate label noise. Test labels are clean.

Baseline: a stand-in for the paper's curation. Within each class, the 20% of training points with the largest PCA reconstruction error are dropped, then an RBF SVM with default C and gamma is trained. This is not the paper's L1-PCA method.

Improved model: an RBF SVM with class_weight='balanced', trained on the full noisy training set, with C and gamma chosen by 5-fold stratified grid search on training data only.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (plus `figures/macro_f1_bar.png`).

## Results (latest run)

On the public credit-g dataset (250-row clean test split, 10% training-label noise), a standard PCA-reconstruction-error curation followed by an RBF SVM reached 64.1% macro-F1, while a class-balanced, CV-tuned RBF SVM on the full training set reached 68.1%, a modest gain of about 4 points.

| Metric | Value |
|---|---|
| baseline_score | 0.6409 |
| baseline_score_std | 0.03461 |
| baseline_accuracy | 0.704 |
| baseline_roc_auc | 0.7346 |
| improved_score | 0.6809 |
| improved_score_std | 0.01173 |
| improved_accuracy | 0.6973 |
| improved_roc_auc | 0.7961 |
| improvement | 0.04008 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Baseline is a stand-in for the paper's L1-PCA curation (L2-PCA proxy); not the paper's method or dataset. Only one stratified split (seed 42); variability reflects label-noise draws (seeds 42-44) only, std is population std over 3 runs. Neither model is randomised at fit time. Test set used once for final scoring. Class for PCA curation uses noisy labels, as in practice. Small test set (250 rows) means noisy macro-F1 estimates.

## Reference

This project is motivated by the following paper:

> S Shukla, DA Pados, G Sklivanitis (2025). *Training Dataset Curation by L1-Norm Principal-Component Analysis for Support Vector Machines*. … on Neural Networks. https://ieeexplore.ieee.org/abstract/document/11010109/

Paper author: Prof. Dimitris Pados, Florida Atlantic University (dpados@fau.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
