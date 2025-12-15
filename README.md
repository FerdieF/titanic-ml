## 🛳 Titanic Survival Prediction — End-to-End ML Pipeline

This repository contains an end-to-end machine learning pipeline for predicting passenger survival on the Titanic dataset.  
The project demonstrates a complete **AI Engineer workflow**, from data preprocessing and model training to model serving via an API.

---

## 🚀 Project Overview

The goal of this project is to build a reproducible and deployable machine learning system that predicts whether a Titanic passenger would survive, based on demographic and travel-related features.

This project includes:

- Data preprocessing and feature engineering
- Model training using a Scikit-learn Pipeline
- Model persistence (saved as artifact)
- Model serving via FastAPI
- Ready-to-use REST API with Swagger documentation

---

## 📂 Dataset

The dataset is sourced from the **Kaggle Titanic Competition**.

Files used:
- `train.csv` — training data
- `test.csv` — inference data (optional)

> ⚠️ Dataset files are not included in this repository.  
> Please download them from Kaggle and place them in the `data/` directory.

---

## 🔧 Preprocessing & Features

### Numerical Features
- `Pclass`
- `Age` (median imputation)
- `SibSp`
- `Parch`
- `Fare` (median imputation)

### Categorical Features
- `Sex`
- `Embarked`

Categorical features are handled using **One-Hot Encoding**, and all preprocessing steps are encapsulated in a Scikit-learn `Pipeline`.

---

## 🤖 Model

- **Algorithm**: Random Forest Classifier
- **Framework**: Scikit-learn
- **Architecture**:
  - ColumnTransformer for preprocessing
  - RandomForestClassifier for prediction

The trained model is saved as:
