# Income classification on census data

Fitting a model to predict whether a person earns more than $50k/year using demographic and employment features (25k rows, 75/25 class imbalance), with scikit-learn pipelines.

Originally a university project for a ML course, later cleaned up and adapted for release.

## Results
Evaluated on a stratified held-out test set:

- Majority class: 0.758 accuracy, 0.000 F1 score
- Decision tree: 0.815 accuracy, 0.610 F1 score
- Random forest + SMOTE (untuned): 0.842 accuracy, 0.667 F1 score
- **Tuned gradient boosting**: 0.832 accuracy, **0.708 F1 score**

The tuned model has a precision of 0.611 and recall of 0.843. Nested 5-fold CV gives F1 = 0.705 ± 0.010.

Accuracy is lower than the untuned forest because the search optimizes for F1. The final model trades precision for recall.

## Approach
- **Preprocessing:** ColumnTransformer (median/mode imputation, merging related categories, one-hot encoding). As it is inside a single pipeline, imputation and encoding are fit only on the training data.
- **Imbalance:** Tuned with random search (none, SMOTE or random oversampling).
- **Model selection:** Randomized search (10 of 1,440 configurations, scored on F1) on scaling, resampling, optional PCA, and random forest vs gradient boosting.
- **Selected model:** Gradient boosting (200 trees, LR = 0.1), random oversampling, PCA with 25 components, no scaling.
- **Evaluation:** Nested cross-validation, then a single eval on the held-out set. Seed is 42 for reproducibility.

## Requirements
The .ipynb has saved cell outputs. To reproduce the results, the requirements are:

`pip install pandas numpy scikit-learn imbalanced-learn matplotlib jupyter`

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/amirhosseinmirzaie/americancitizenincome), and must be saved as `data/income.csv`.