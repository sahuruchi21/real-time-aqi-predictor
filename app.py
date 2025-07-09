
import streamlit as st
import numpy as np
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Streamlit config
st.set_page_config(page_title="Real-Time AQI Forecast", layout="centered")
st.title("🌫️ Real-Time AQI Forecast (Next 15 Days)")
st.markdown("""Predict air quality using live AQI from WAQI API and an LSTM model.""")

# Load the model
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_aqi_model.h5")

# Load historical data
@st.cache_data
def load_historical_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['City'] == 'Delhi'].sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df = df[['AQI']].interpolate().dropna()
    return df

# Prepare scaler and sequence
def prepare_scaler_and_sequence(df, n_steps=30):
    scaler = MinMaxScaler()
    aqi_scaled = scaler.fit_transform(df[['AQI']].values)
    last_sequence = aqi_scaled[-n_steps:].reshape(1, n_steps, 1)
    return scaler, last_sequence

# Get live AQI from WAQI
def fetch_live_aqi_waqi(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url).json()
    if response["status"] != "ok":
        return None, "City not found or API error."
    try:
        return response["data"]["aqi"], None
    except:
        return None, "AQI data not available."

# AQI Category with Emoji
def aqi_label(aqi):
    if aqi <= 50:
        return "😊 Good"
    elif aqi <= 100:
        return "🙂 Satisfactory"
    elif aqi <= 200:
        return "😐 Moderate"
    elif aqi <= 300:
        return "😷 Unhealthy"
    elif aqi <= 400:
        return "🤢 Very Unhealthy"
    else:
        return "☠️ Hazardous"

# Advisory Sections
def health_advisory(aqi):
    if aqi <= 100:
        return "🟢 Air is acceptable. No precautions needed."
    elif aqi <= 200:
        return "⚠️ Sensitive groups should limit prolonged outdoor exertion."
    elif aqi <= 300:
        return "🚑 Avoid outdoor activity. Respiratory issues may worsen."
    else:
        return "🚨 Health emergency conditions. Stay indoors."

def farming_advisory(aqi):
    if aqi <= 100:
        return "🌾 Farming activities are safe."
    elif aqi <= 200:
        return "📅 Reduce exposure during spraying/harvesting."
    else:
        return "❌ Delay outdoor agricultural tasks."

def transport_advisory(aqi):
    if aqi <= 100:
        return "🚌 Use public transport to reduce pollution."
    elif aqi <= 200:
        return "🚕 Consider carpooling and reduce travel time."
    else:
        return "⛔ Restrict vehicle movement in sensitive zones."

def research_recommendation(aqi):
    if aqi > 200:
        return "🔬 Correlate with local hospital admissions for health studies."
    return ""

# Forecast next N days
def forecast_aqi(model, scaler, last_seq, future_days=15):
    preds_scaled = []
    seq = last_seq.copy()
    for _ in range(future_days):
        next_pred = model.predict(seq, verbose=0)[0]
        preds_scaled.append(next_pred)
        seq = np.append(seq[:, 1:, :], [[next_pred]], axis=1)
    preds = scaler.inverse_transform(preds_scaled)
    future_dates = pd.date_range(datetime.date.today() + datetime.timedelta(days=1), periods=future_days)
    return pd.DataFrame({'Date': future_dates, 'Predicted_AQI': preds.flatten()})

# Load model and data
model = load_lstm_model()
df = load_historical_data()

# UI
st.subheader("💾 Input City & WAQI API Token")
city = st.text_input("Enter City Name (India)", value="Delhi")
waqi_token = st.text_input("Enter your WAQI API Token", type="password")

# Run prediction
if st.button("Fetch & Predict"):
    if not waqi_token:
        st.error("⚠️ WAQI token is required.")
    else:
        live_aqi, error = fetch_live_aqi_waqi(city, waqi_token)
        if error:
            st.error(error)
        else:
            st.success(f"✅ Live AQI in {city}: {live_aqi}")
            scaler, last_seq = prepare_scaler_and_sequence(df, n_steps=30)
            new_scaled = scaler.transform([[live_aqi]])
            updated_seq = np.append(last_seq[:, 1:, :], [[new_scaled[0]]], axis=1)
            forecast_df = forecast_aqi(model, scaler, updated_seq)

            # Add all labels and advisories
            forecast_df["Category"] = forecast_df["Predicted_AQI"].apply(aqi_label)
            forecast_df["AQI with Label"] = forecast_df.apply(
                lambda row: f"{row['Predicted_AQI']:.2f} {row['Category']}", axis=1
            )
            forecast_df["Health Advisory"] = forecast_df["Predicted_AQI"].apply(health_advisory)
            forecast_df["Farming Advisory"] = forecast_df["Predicted_AQI"].apply(farming_advisory)
            forecast_df["Transport Advisory"] = forecast_df["Predicted_AQI"].apply(transport_advisory)
            forecast_df["Research Hint"] = forecast_df["Predicted_AQI"].apply(research_recommendation)

            # Display data table
            st.subheader("📊 Predicted AQI for Next 15 Days")
            st.dataframe(forecast_df[[
                "Date", "AQI with Label", "Health Advisory",
                "Farming Advisory", "Transport Advisory", "Research Hint"
            ]].style.highlight_max(subset=["AQI with Label"], color='lightcoral'))

            # Warning
            max_aqi = forecast_df["Predicted_AQI"].max()
            if max_aqi > 400:
                st.error("☠️ AQI will reach hazardous levels. Stay indoors!")
            elif max_aqi > 300:
                st.error("🤢 Very Unhealthy air expected. Avoid outdoor activity!")
            elif max_aqi > 200:
                st.warning("😷 Unhealthy air expected. Limit exposure.")
            elif max_aqi > 100:
                st.info("🙂 AQI moderate to unhealthy for sensitive groups.")
            else:
                st.success("😊 AQI remains in a healthy range.")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_df['Date'], forecast_df['Predicted_AQI'], marker='o', color='orange', label='Forecast')
            ax.set_title(f"AQI Forecast for {city}")
            ax.set_xlabel("Date")
            ax.set_ylabel("AQI")
            ax.grid(True)
            ax.legend()
            plt.xticks(rotation=45)

            # Annotate peak AQI
            max_row = forecast_df.loc[forecast_df["Predicted_AQI"].idxmax()]
            ax.annotate(
                aqi_label(max_row["Predicted_AQI"]),
                xy=(max_row["Date"], max_row["Predicted_AQI"]),
                xytext=(max_row["Date"], max_row["Predicted_AQI"] + 10),
                ha='center',
                color='red',
                fontsize=12,
                arrowprops=dict(facecolor='red', shrink=0.05)
            )

            st.pyplot(fig)
