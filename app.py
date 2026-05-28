import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from utils import load_data, FEATURES, get_aqi_bucket


st.set_page_config(
    page_title="Global AQI Dashboard",
    layout="wide"
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return joblib.load("aqi_model.pkl")


df = get_data()
model = get_model()


st.title("🌍 Global Air Quality Dashboard")
st.write("EDA, pollutant trends, AQI classification, and AQI prediction using machine learning.")


# Sidebar
st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Select Country",
    sorted(df["Country"].unique())
)

country_df = df[df["Country"] == country]

city = st.sidebar.selectbox(
    "Select City",
    sorted(country_df["City"].unique())
)

city_df = country_df[country_df["City"] == city]


# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Countries", df["Country"].nunique())
col3.metric("Cities", df["City"].nunique())
col4.metric("Selected City Records", len(city_df))


# AQI trend
st.subheader(f"AQI Trend: {city}, {country}")

fig_aqi = px.line(
    city_df,
    x="Date",
    y="AQI",
    title=f"AQI Over Time in {city}"
)

st.plotly_chart(fig_aqi, use_container_width=True)


# Pollutant trend
st.subheader("Pollutant Trends")

pollutants = [
    "PM2.5 (ug/m3)",
    "PM10 (ug/m3)",
    "NO (ug/m3)",
    "NO2 (ug/m3)",
    "NOx (ppb)",
    "NH3 (ug/m3)",
    "CO (mg/m3)",
    "SO2 (ug/m3)",
    "O3 (ug/m3)",
    "Benzene (ug/m3)",
    "Toluene (ug/m3)",
    "Xylene (ug/m3)"
]
selected_pollutant = st.selectbox(
    "Select Pollutant",
    pollutants
)

fig_pollutant = px.line(
    city_df,
    x="Date",
    y=selected_pollutant,
    title=f"{selected_pollutant} Trend in {city}"
)

st.plotly_chart(fig_pollutant, use_container_width=True)


# AQI bucket distribution
st.subheader("AQI Category Distribution")

bucket_counts = city_df["AQI_Bucket"].value_counts().reset_index()
bucket_counts.columns = ["AQI_Bucket", "Count"]

fig_bucket = px.bar(
    bucket_counts,
    x="AQI_Bucket",
    y="Count",
    title=f"AQI Bucket Distribution in {city}"
)

st.plotly_chart(fig_bucket, use_container_width=True)


# Country comparison
st.subheader("Average AQI by City")

city_avg = (
    country_df
    .groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig_city_avg = px.bar(
    city_avg,
    x="City",
    y="AQI",
    title=f"Average AQI by City in {country}"
)

st.plotly_chart(fig_city_avg, use_container_width=True)


# Correlation heatmap
st.subheader("Feature Correlation Heatmap")

corr_cols = FEATURES + ["AQI"]
corr = df[corr_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=False,
    title="Correlation Between Features and AQI"
)

st.plotly_chart(fig_corr, use_container_width=True)


# Feature importance
st.subheader("Model Feature Importance")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Random Forest Feature Importance"
)

st.plotly_chart(fig_importance, use_container_width=True)


# AQI prediction
st.subheader("Predict AQI")

st.write("Enter pollutant, weather, and environmental values to predict AQI.")

input_data = {}

for feature in FEATURES:
    input_data[feature] = st.number_input(
        feature,
        value=float(df[feature].mean())
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict AQI"):
    predicted_aqi = model.predict(input_df)[0]
    predicted_bucket = get_aqi_bucket(predicted_aqi)

    st.success(f"Predicted AQI: {round(predicted_aqi, 2)}")
    st.info(f"Predicted AQI Category: {predicted_bucket}")