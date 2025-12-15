import os
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = os.path.join("models", "model.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🛳 Titanic Survival Prediction")
st.write(
    "Simple UI to predict whether a passenger would survive the Titanic disaster "
    "using a trained RandomForest model."
)

st.sidebar.header("Passenger Features")

# Input fields
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", min_value=0, max_value=80, value=22)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=1)
parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=600.0, value=7.25, step=0.25)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"], index=2)

if st.button("Predict Survival"):
    # Buat DataFrame dengan kolom yang sama seperti saat training
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Embarked": embarked,
    }])

    proba = model.predict_proba(input_data)[0][1]
    pred = int(proba >= 0.5)

    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"✅ The model predicts that this passenger **would survive** "
                   f"(probability: {proba:.2%}).")
    else:
        st.error(f"❌ The model predicts that this passenger **would not survive** "
                 f"(probability of survival: {proba:.2%}).")

    st.markdown("**Input summary:**")
    st.json(input_data.to_dict(orient="records")[0])
