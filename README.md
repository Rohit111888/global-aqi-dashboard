# 🌍 Global AQI Dashboard

An interactive air quality dashboard built using Python, Streamlit, Pandas, Plotly, and Scikit-learn.

## Features

- AQI trend visualization
- Country and city filters
- Pollutant trend analysis
- AQI category distribution
- Machine learning AQI prediction
- Feature importance visualization

## Tech Stack

- Python
- Pandas
- Streamlit
- Plotly
- Scikit-learn
- Random Forest Regressor

## Dataset

Global Air Quality Dataset (2014–2025)

Includes:
- 24 countries
- 300k+ records
- AQI and pollutant measurements
- Weather and environmental indicators

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python model.py
```

Run dashboard:

```bash
streamlit run app.py
```

## Project Structure

```text
global-aqi-dashboard/
│
├── Data/
├── app.py
├── model.py
├── utils.py
├── requirements.txt
└── README.md
```

## Future Improvements

- LSTM forecasting
- Geospatial AQI maps
- SHAP explainability
- Live API integration
- Streamlit Cloud deployment