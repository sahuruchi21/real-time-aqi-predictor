
# 🌫️ Real-Time AQI Forecasting Web App

### 🔗 [Live Demo](https://your-streamlit-cloud-link) &emsp;|&emsp; 🧠 Powered by LSTM & Streamlit

---

## 📌 Description
This web application forecasts Air Quality Index (AQI) for the next 15 days for any Indian city using real-time data from the **WAQI API** and a trained **LSTM deep learning model**. It provides smart **health, agriculture, transport**, and **research** advisories based on predicted AQI levels.

---

## 🚀 Features
- ✅ Real-time AQI fetching via WAQI API
- 📈 15-day AQI Forecast using LSTM model
- 🌡️ Health, Transport & Farming advisories
- 🧪 Research recommendation system
- 🖼️ Interactive charts with forecast annotation
- 🔐 Secure token handling
- ☁️ Deployable on Streamlit Cloud

---

## 🖥️ Tech Stack
| Component     | Tool              |
|---------------|-------------------|
| Web Framework | `Streamlit`       |
| ML Model      | `LSTM (Keras)`    |
| Data Source   | `WAQI API`        |
| Visualization | `Matplotlib`      |
| Scaler        | `MinMaxScaler`    |
| Language      | `Python`          |

---

## 🛠️ Installation
```bash
git clone https://github.com/yourusername/aqi-forecast-app.git
cd aqi-forecast-app
pip install -r requirements.txt
```

---

## 🧠 Model Info
- Trained on Delhi AQI data (`city_day.csv`)
- Used LSTM with 30 time steps
- Includes multiple LSTM layers and Dropout regularization
- Saved as `lstm_aqi_model.h5`

---

## 🔐 WAQI API Setup
1. Create a free token from [https://aqicn.org/data-platform/token](https://aqicn.org/data-platform/token)
2. Add it inside the app or as an environment variable.

---

## ▶️ Run Locally
```bash
streamlit run app.py
```

---

## 📦 File Structure
```
📁 aqi-forecast-app/
├── app.py                # Streamlit app
├── lstm_aqi_model.h5     # Trained LSTM model
├── city_day.csv          # AQI dataset for scaling
├── requirements.txt
└── README.md
```

---

## 📊 Sample Output Screenshots

### 1. Dashboard Home Page

![Dashboard Home](https://drive.google.com/file/d/1K_hlflGTv3CXqIAH01ZO0a29Xm8n-dKK/view?usp=drive_link)

---

### 2. AQI Prediction Results

![AQI Prediction](https://drive.google.com/file/d/1NowCI8aePlbCEL54R71poB6JivALSLea/view?usp=drive_link)

---

### 3. Pollutant Trends Visualization

![Pollutant Trends](https://drive.google.com/file/d/1NowCI8aePlbCEL54R71poB6JivALSLea/view?usp=drive_link)


---

## 📌 Future Enhancements
- Add more features like humidity, temperature, wind
- Geo-based smart notifications
- Real-time model retraining pipeline

---

## 🤝 Acknowledgments
- [WAQI API](https://waqi.info)
- [Kaggle Air Quality Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- Streamlit Community
