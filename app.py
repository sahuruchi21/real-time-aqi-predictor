import sys
import streamlit as st

st.write("Python version:", sys.version)

import streamlit as st
import numpy as np
import pandas as pd
import requests
import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from streamlit_autorefresh import st_autorefresh  

# 🔁 Auto-refresh every 15 minutes
st_autorefresh(interval=900000, key="auto_refresh")

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

# Get live AQI + pollutants from WAQI
def fetch_live_aqi_waqi(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url).json()
    if response["status"] != "ok":
        return None, {}, "City not found or API error."
    try:
        aqi = response["data"]["aqi"]
        iaqi = response["data"].get("iaqi", {})  
        pollutants = {k: v.get("v", None) for k, v in iaqi.items()}
        return aqi, pollutants, None
    except:
        return None, {}, "AQI data not available."

# AQI Category with Emoji
def aqi_label(aqi):
    if aqi <= 50: return "😊 Good"
    elif aqi <= 100: return "🙂 Satisfactory"
    elif aqi <= 200: return "😐 Moderate"
    elif aqi <= 300: return "😷 Unhealthy"
    elif aqi <= 400: return "🤢 Very Unhealthy"
    else: return "☠️ Hazardous"

# Advisory Sections
def health_advisory(aqi):
    if aqi <= 100: return "🟢 Air is acceptable. No precautions needed."
    elif aqi <= 200: return "⚠️ Sensitive groups should limit prolonged outdoor exertion."
    elif aqi <= 300: return "🚑 Avoid outdoor activity. Respiratory issues may worsen."
    else: return "🚨 Health emergency conditions. Stay indoors."

def farming_advisory(aqi):
    if aqi <= 100: return "🌾 Farming activities are safe."
    elif aqi <= 200: return "📅 Reduce exposure during spraying/harvesting."
    else: return "❌ Delay outdoor agricultural tasks."

def transport_advisory(aqi):
    if aqi <= 100: return "🚌 Use public transport to reduce pollution."
    elif aqi <= 200: return "🚕 Consider carpooling and reduce travel time."
    else: return "⛔ Restrict vehicle movement in sensitive zones."

def research_recommendation(aqi):
    if aqi <= 100: return "📈 Monitor trends to identify early pollution patterns."
    elif aqi <= 150: return "🔍 Study correlation with respiratory outpatient cases."
    elif aqi <= 200: return "🧬 Analyze pollutant composition for dominant contributors (PM, NOx, SO₂)."
    elif aqi <= 300: return "🏥 Compare AQI with hospital admission and emergency room visits."
    elif aqi <= 400: return "📊 Model health-economic impact from reduced outdoor workforce efficiency."
    else: return "🚨 Urgent: Recommend emergency response policies and shelter-in-place drills."

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
        live_aqi, pollutants, error = fetch_live_aqi_waqi(city, waqi_token)
        if error:
            st.error(error)
        else:
            st.success(f"✅ Live AQI in {city}: {live_aqi} ({aqi_label(live_aqi)})")

            # --- Pollutant Breakdown ---
            if pollutants:
                st.subheader("🧪 Pollutant Breakdown")
                pol_df = pd.DataFrame(list(pollutants.items()), columns=["Pollutant", "Value"])
                fig_pol = px.bar(pol_df, x="Pollutant", y="Value", color="Value",
                                 title=f"Pollutant Levels in {city}",
                                 color_continuous_scale="Viridis")
                st.plotly_chart(fig_pol, use_container_width=True)

            # --- Forecast ---
            scaler, last_seq = prepare_scaler_and_sequence(df, n_steps=30)
            new_scaled = scaler.transform([[live_aqi]])
            updated_seq = np.append(last_seq[:, 1:, :], [[new_scaled[0]]], axis=1)
            forecast_df = forecast_aqi(model, scaler, updated_seq)

            # Add labels + advisories
            forecast_df["Category"] = forecast_df["Predicted_AQI"].apply(aqi_label)
            forecast_df["AQI with Label"] = forecast_df.apply(
                lambda row: f"{row['Predicted_AQI']:.2f} {row['Category']}", axis=1
            )
            forecast_df["Health Advisory"] = forecast_df["Predicted_AQI"].apply(health_advisory)
            forecast_df["Farming Advisory"] = forecast_df["Predicted_AQI"].apply(farming_advisory)
            forecast_df["Transport Advisory"] = forecast_df["Predicted_AQI"].apply(transport_advisory)
            forecast_df["Research Hint"] = forecast_df["Predicted_AQI"].apply(research_recommendation)

            # --- Historical + Forecast Plot (30 days) ---
            st.subheader("📈 Historical & Forecasted AQI")
            hist_df = df[-30:].reset_index()  
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['AQI'],
                                          mode='lines+markers', name="Historical AQI", line=dict(color="blue")))
            fig_hist.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_AQI'],
                                          mode='lines+markers', name="Forecast AQI", line=dict(color="orange")))
            fig_hist.update_layout(title=f"AQI Trend & Forecast for {city}",
                                   xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Real-Time Alerts ---
            if live_aqi > 400:
                st.error("☠️ Hazardous air quality right now! Stay indoors!")
            elif live_aqi > 300:
                st.error("🤢 Very Unhealthy air quality detected. Avoid outdoor activity!")
            elif live_aqi > 200:
                st.warning("😷 Unhealthy air quality. Sensitive groups at risk.")
            elif live_aqi > 100:
                st.info("🙂 Moderate AQI. Sensitive groups should limit exposure.")
            else:
                st.success("😊 Air quality is good today!")

            # --- Forecast Table ---
            st.subheader("📊 Predicted AQI for Next 15 Days with Advisories")
            st.dataframe(forecast_df[[
                "Date", "AQI with Label", "Health Advisory",
                "Farming Advisory", "Transport Advisory", "Research Hint"
            ]])

            # --- Download Option ---
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv,
                file_name=f"AQI_Forecast_{city}.csv",
                mime="text/csv"
            )
