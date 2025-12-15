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
models/model.pkl


---

## 🧠 Machine Learning Pipeline

The pipeline includes:
1. Missing value handling
2. Feature encoding
3. Model training
4. Validation
5. Model serialization

--

## 🌐 API (FastAPI)

The trained model is served using FastAPI.

Run the API
uvicorn app.main:app --reload

API Endpoints

GET /
Health check

POST /predict
Predict passenger survival

Example Request
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

Example Response
```
{
  "input": {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  },
  "survived": 0,
  "probability_survived": 0.1033
}
```

Swagger UI is available at:

`http://127.0.0.1:8000/docs`

## 📁 Project Structure
```
titanic-ml/
│
├── app/
│   └── main.py              # FastAPI app
│
├── data/
│   └── train.csv            # (not included)
│
├── models/
│   └── model.pkl            # trained model
│
├── titanic_ml/
│   ├── __init__.py
│   └── pipeline.py          # ML pipeline definition
│
├── train.py                 # training script
├── requirements.txt
└── README.md
```

## 🧑‍💻 How to Run Locally
```
# clone repository
git clone https://github.com/FerdieF/titanic-ml.git
cd titanic-ml

# create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# install dependencies
pip install -r requirements.txt

# train model
python train.py

# start API
uvicorn app.main:app --reload
```

## 📈 Future Improvements
- Add advanced feature engineering (Title, FamilySize, IsAlone)
- Model comparison (XGBoost, CatBoost)
- Dockerization
- CI/CD pipeline
- Deployment to cloud platforms (Render, Railway, HuggingFace Spaces)

## 👤 Author
**FerdieF**
Machine Learning / AI Engineering Portfolio Project
