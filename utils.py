import pandas as pd


FEATURES = [
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
    "Xylene (ug/m3)",
    "Wind_Speed (km/h)",
    "Humidity (%)",
    "Deforestation_Rate_%",
    "Industry_Growth_%",
    "CO2_Emission_MT",
    "Population_Density_per_SqKm"
]

def load_data(path="data/global_air_quality_2014_2025.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def get_aqi_bucket(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 150:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    elif aqi <= 300:
        return "Very Poor"
    else:
        return "Severe"