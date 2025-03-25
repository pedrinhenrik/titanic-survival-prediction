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

## Exploratory Data Analysis

Before building the model, an exploratory data analysis (EDA) was performed to understand the structure and distribution of the dataset. This step helps identify missing values, outliers, correlations, and patterns that may influence survival.

### Age Distribution by Survival

![Age Distribution](images/age_distribution.png)

The distribution shows that younger passengers had a higher chance of survival. This aligns with the prioritization of children during evacuation.

### Survival Rate by Gender

![Survival by Sex](images/survival_by_sex.png)

Female passengers had a significantly higher survival rate compared to males, which supports historical accounts that women and children were prioritized during lifeboat boarding.
