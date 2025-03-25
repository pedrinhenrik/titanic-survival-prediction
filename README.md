# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using real historical data. This project includes data exploration, feature engineering, and model building using a complete Scikit-learn pipeline with hyperparameter tuning via cross-validation.

---

## Project Overview

This repository demonstrates how to use structured data to train a classification model that predicts whether a passenger would have survived the Titanic disaster. In addition to building a predictive model, this project extracts meaningful insights from the data, reflecting real-world patterns such as age, gender, and social class.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Insights and Conclusions](#insights-and-conclusions)

## Modeling

To predict passenger survival, a supervised classification approach was implemented using a **Random Forest Classifier**, integrated into a complete machine learning pipeline. The pipeline includes:

- Handling missing values (imputation)
- Feature scaling for numerical features
- One-hot encoding for categorical variables
- Hyperparameter tuning using `GridSearchCV`

The dataset was split into **80% training** and **20% testing**, maintaining class balance with `stratify=y`. Cross-validation was performed using **StratifiedKFold (5 folds)** to ensure stability in the evaluation.

### Model Used
- **Algorithm:** Random Forest Classifier
- **Cross-validation:** 5-fold StratifiedKFold
- **Hyperparameters Tuned:**  
  - `n_estimators` = 50, 100  
  - `max_depth` = None, 10, 20  
  - `min_samples_split` = 2, 5

### Best Parameters Found
```python
{'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}

## Exploratory Data Analysis

Before building the model, an exploratory data analysis (EDA) was performed to understand the structure and distribution of the dataset. This step helps identify missing values, outliers, correlations, and patterns that may influence survival.

### Age Distribution by Survival

![Age Distribution](images/age_distribution.png)

The distribution shows that younger passengers had a higher chance of survival. This aligns with the prioritization of children during evacuation.

### Survival Rate by Gender

![Survival by Gender](images/survival_by_sex.png)

Female passengers had a significantly higher survival rate compared to males, which supports historical accounts that women and children were prioritized during lifeboat boarding.
