import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


NUM_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CAT_FEATURES = ["Sex", "Embarked"]


def build_model(random_state: int = 42):
    """
    Build sklearn Pipeline: preprocessing + classifier.
    """
    # Numerical: impute median
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical: impute most frequent + one-hot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=4,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    return model


def get_features_target(df: pd.DataFrame):
    """
    Split dataframe into X, y for training.
    Assumes there is a 'Survived' column.
    """
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    y = df["Survived"].astype(int)
    return X, y
