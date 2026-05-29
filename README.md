# Manufacturing Defect Prediction

A machine learning pipeline to predict manufacturing defects (`DefectStatus`) using classification models trained on the `manufacturing\\\_defect\\\_dataset.csv` dataset.

\---

## Project Overview

This project builds and evaluates multiple classification models to predict whether a manufacturing product is defective or not. It covers the full ML workflow: data loading, EDA, preprocessing, handling class imbalance, model training, and evaluation.

\---

## Dataset

**File:** `manufacturing\\\_defect\\\_dataset.csv`

**Target Variable:** `DefectStatus` (binary — defective or not)

**Key Features:**

|Feature|Description|
|-|-|
|`ProductionVolume`|Volume of units produced|
|`ProductionCost`|Cost of production|
|`DefectRate`|Rate of defects observed|
|`QualityScore`|Quality control score|
|`MaintenanceHours`|Hours spent on maintenance|
|`WorkerProductivity`|Worker productivity metric|
|`EnergyConsumption`|Energy used in production|
|`AdditiveMaterialCost`|Cost of additive materials|

\---

## Project Structure

```
manufacturing-defect-prediction/
│
├── manufacturing\\\_defect\\\_dataset.csv   # Input dataset
├── defect.ipynb                       # Main Jupyter Notebook
└── README.md                          # Project documentation
```

\---

## Installation

1. **Clone or download** this repository.
2. **Install required Python packages:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost statsmodels
```

3. **Place the dataset** (`manufacturing\\\_defect\\\_dataset.csv`) in the same directory as the notebook.
4. **Launch Jupyter Notebook:**

```bash
jupyter notebook defect.ipynb
```

\---

## Usage

Run all cells in the notebook sequentially from top to bottom. Each section is clearly labelled with markdown headings.

\---

## Pipeline Walkthrough

### 1\. Data Loading \& Inspection

* Load dataset with `pandas`
* Preview with `df.head()` and `df.info()`

### 2\. Data Quality Checks

* Check for missing values (`df.isnull().sum()`)
* Check for duplicate rows (`df.duplicated().sum()`)

### 3\. Outlier Detection

* Uses the **Interquartile Range (IQR)** method on all numeric features (excluding the target)
* Reports the number of outliers per column

### 4\. Exploratory Data Analysis (EDA)

* **Histograms** for 8 key features to visualise distributions
* **Correlation Heatmap** to identify relationships between variables

### 5\. Multicollinearity Check

* Computes **Variance Inflation Factor (VIF)** for all features
* Helps identify redundant or highly correlated predictors

### 6\. Preprocessing

* Splits features (`X`) and target (`y`)
* **Train/Test Split:** 80% training, 20% testing (`random\\\_state=42`)
* **Feature Scaling:** `StandardScaler` applied to training and test sets

### 7\. Class Imbalance Handling

* Visualises `DefectStatus` class distribution
* Applies **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training set

### 8\. Model Training \& Evaluation

Four classifiers are trained on the SMOTE-resampled data and evaluated on the held-out test set.

\---

## Models

|Model|Key Notes|
|-|-|
|Logistic Regression|Solver: `liblinear`; suitable for smaller datasets|
|Random Forest|Ensemble of decision trees; handles non-linearity well|
|Decision Tree|Single tree; interpretable but prone to overfitting|
|XGBoost|Gradient-boosted trees; `eval\\\_metric='logloss'`|

\---

## Results

Each model is evaluated using Accuracy, Precision, Recall, F1-Score, Classification Report, and a Confusion Matrix heatmap.

|Model|Accuracy|
|-|-|
|Logistic Regression|76.85%|
|Decision Tree|88.58%|
|Random Forest|**95.52%**|
|XGBoost|**95.52%**|

Random Forest and XGBoost are the top-performing models, both achieving the same accuracy of 95.52%. Decision Tree offers a reasonable middle ground, while Logistic Regression lags behind — likely due to the non-linear nature of the data.

\---

## 


