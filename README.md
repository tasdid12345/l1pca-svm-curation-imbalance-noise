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

## Reference

This project is motivated by the following paper:

> S Shukla, DA Pados, G Sklivanitis (2025). *Training Dataset Curation by L1-Norm Principal-Component Analysis for Support Vector Machines*. … on Neural Networks. https://ieeexplore.ieee.org/abstract/document/11010109/

Paper author: Prof. Dimitris Pados, Florida Atlantic University (dpados@fau.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
