import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Smart Agriculture Analytics",
    page_icon="🌱",
    layout="wide"
)

MODEL_DIR = Path(__file__).parent / "stage26_models"

@st.cache_resource
def load_models():
    crop_model = joblib.load(MODEL_DIR / "crop_classifier.joblib")
    yield_model = joblib.load(MODEL_DIR / "yield_predictor.joblib")
    return crop_model, yield_model

try:
    crop_model, yield_model = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.title("🌱 Smart Agriculture Analytics")
st.write("Crop recommendation and experimental yield prediction")

page = st.sidebar.selectbox(
    "Select a section",
    ["Crop Recommendation", "Yield Prediction", "Model Insights"]
)

if page == "Crop Recommendation":

    st.header("Crop Recommendation")
    st.write("Enter the 12 model input features.")

    features = [
        "humidity", "K", "temperature", "N", "P_K_Ratio",
        "NPK_Total", "Temperature_Rainfall", "rainfall",
        "N_K_Ratio", "Humidity_Rainfall",
        "Temperature_Humidity", "P"
    ]

    values = {}

    for feature in features:
        values[feature] = st.number_input(feature, value=0.0)

    if st.button("Recommend Crop"):
        X = pd.DataFrame([values], columns=features)
        prediction = crop_model.predict(X)[0]
        st.success(f"Recommended crop: {prediction}")

elif page == "Yield Prediction":

    st.header("Yield Prediction")
    st.warning(
        "This is an experimental model. It did not meaningfully "
        "outperform the Dummy baseline. Yield units are unverified."
    )

    features = [
        "NDVI_Temperature", "OrganicMatter_Sand",
        "Temperature_Rainfall", "sand_pct", "NDVI_Rainfall",
        "soil_ph", "organic_matter", "Humidity_Rainfall",
        "Temperature_Humidity", "rainfall"
    ]

    values = {}

    for feature in features:
        values[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Yield"):
        X = pd.DataFrame([values], columns=features)
        prediction = yield_model.predict(X)[0]
        st.metric("Predicted yield (unit unknown)", f"{prediction:.2f}")

else:

    st.header("Model Insights")

    st.subheader("Crop Classification")
    st.write("Earlier Random Forest 5-fold CV accuracy: 98.92%")
    st.write("Preliminary XGBoost holdout accuracy: 99.55%")

    st.warning(
        "The crop holdout result is preliminary. Feature-selection "
        "leakage has not yet been ruled out."
    )

    st.subheader("Yield Prediction")
    st.write("Dummy baseline MAE: 378.89")
    st.write("Tuned Gradient Boosting MAE: 379.05")

    st.info(
        "The yield model did not meaningfully beat the baseline. "
        "Its predictions should not be treated as reliable estimates."
    )
