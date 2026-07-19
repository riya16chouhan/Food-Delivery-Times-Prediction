import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        padding: 1.2rem 1.5rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
    }
    .result-time {
        font-size: 2.4rem;
        font-weight: 800;
        color: #047857;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🛵 Food Delivery Time Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict how long a delivery will take, using a Random Forest model trained on historical orders.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model (cached so it's only loaded once per session)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model (1).pkl")


model = load_model()

# The exact columns the model was trained on (order matters for a plain numpy
# array, but since we build a DataFrame with these names, sklearn will align
# on column name instead of position).
MODEL_COLUMNS = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Weather_junk",  # artifact column from the original training notebook, always 0
    "Weather_Clear",
    "Weather_Foggy",
    "Weather_Rainy",
    "Weather_Snowy",
    "Weather_Windy",
    "Traffic_Level_junk",  # artifact column, always 0
    "Traffic_Level_High",
    "Traffic_Level_Low",
    "Traffic_Level_Medium",
    "Time_of_Day_junk",  # artifact column, always 0
    "Time_of_Day_Afternoon",
    "Time_of_Day_Evening",
    "Time_of_Day_Morning",
    "Time_of_Day_Night",
    "Vehicle_Type_Bike",
    "Vehicle_Type_Car",
    "Vehicle_Type_Scooter",
]


def build_feature_row(distance, prep_time, experience, weather, traffic, time_of_day, vehicle):
    row = {col: 0 for col in MODEL_COLUMNS}
    row["Distance_km"] = distance
    row["Preparation_Time_min"] = prep_time
    row["Courier_Experience_yrs"] = experience
    row[f"Weather_{weather}"] = 1
    row[f"Traffic_Level_{traffic}"] = 1
    row[f"Time_of_Day_{time_of_day}"] = 1
    row[f"Vehicle_Type_{vehicle}"] = 1

    df = pd.DataFrame([row])[MODEL_COLUMNS]

    # The model was fit with the raw column names produced by pd.get_dummies,
    # which include two odd artifact columns from a preprocessing bug in the
    # original training notebook. We rename to those exact names right before
    # prediction so sklearn's feature-name check passes.
    rename_map = {
        "Weather_junk": [c for c in model.feature_names_in_ if c.startswith("Weather_<bound")][0],
        "Traffic_Level_junk": [c for c in model.feature_names_in_ if c.startswith("Traffic_Level_<bound")][0],
        "Time_of_Day_junk": [c for c in model.feature_names_in_ if c.startswith("Time_of_Day_<bound")][0],
    }
    df = df.rename(columns=rename_map)
    df = df[model.feature_names_in_]
    return df


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Order details")

    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        prep_time = st.number_input("Preparation time (min)", min_value=0, max_value=120, value=15, step=1)
        experience = st.number_input("Courier experience (years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)

    with col2:
        weather = st.selectbox("Weather", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
        traffic = st.selectbox("Traffic level", ["Low", "Medium", "High"])
        time_of_day = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"])
        vehicle = st.selectbox("Vehicle type", ["Bike", "Scooter", "Car"])

    submitted = st.form_submit_button("Predict delivery time", use_container_width=True)

if submitted:
    features = build_feature_row(distance, prep_time, experience, weather, traffic, time_of_day, vehicle)
    prediction = model.predict(features)[0]

    st.markdown(
        f"""
        <div class="result-box">
            Estimated delivery time<br>
            <span class="result-time">{prediction:.0f} minutes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See the exact inputs sent to the model"):
        st.dataframe(features.T.rename(columns={0: "value"}))

st.markdown("---")
st.caption("Model: Random Forest Regressor (tuned via GridSearchCV) · Test R² ≈ 0.81 · Test MAE ≈ 6.6 min")