
## 📌 Description
This web app forecasts the **Air Quality Index (AQI)** for the next 15 days for Indian cities using real-time data from the **WAQI API** and a pre-trained **LSTM deep learning model**.  
It also provides smart **health, agriculture, transport**, and **research** advisories based on the predicted AQI levels.

---

## 🚀 Features
- ✅ Fetches **real-time AQI** via WAQI API  
- 📈 **15-day AQI forecast** using an LSTM model  
- 🌡️ **Health, Transport & Farming advisories** based on AQI  
- 🧪 **Research recommendations** for data-driven insights  
- 📊 Interactive **visualizations** (Plotly)  
- 🔐 **Secure API token** input  
- ☁️ **Easily deployable** on Streamlit Cloud  

---

## 🖥️ Tech Stack
| Component     | Tool/Library         |
|---------------|-----------------------|
| Web Framework | `Streamlit`           |
| ML Model      | `LSTM (TensorFlow/Keras)` |
| Data Source   | `WAQI API`            |
| Visualization | `Plotly`              |
| Scaler        | `MinMaxScaler`        |
| Language      | `Python`              |

---

## 🛠️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/sahuruchi21/real-time-aqi-predictor.git
cd real-time-aqi-predictor
pip install -r requirements.txt
🧠 Model Info
Trained on Delhi AQI dataset (city_day.csv)

Sequence length: 30 time steps

Model: Stacked LSTM layers with Dropout

Saved model: lstm_aqi_model.h5

🔐 WAQI API Setup
Get a free token from https://aqicn.org/data-platform/token

Enter the token inside the app when prompted.

▶️ Run Locally

streamlit run app.py
📦 File Structure
graphql
Copy code
📁 real-time-aqi-predictor/
├── app.py                # Main Streamlit app
├── lstm_aqi_model.h5     # Trained LSTM model
├── city_day.csv          # AQI dataset for scaling
├── requirements.txt      # Dependencies
├── train_model.py        # Model training script
└── README.md             # Project documentation

## 🤝 Acknowledgments
- [WAQI API](https://waqi.info)
- [Kaggle Air Quality Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- Streamlit Community
