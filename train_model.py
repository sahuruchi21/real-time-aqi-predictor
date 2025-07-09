# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
df = pd.read_csv("city_day.csv")
df = df[df['City'] == 'Delhi'].sort_values(by='Date')
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'AQI']].dropna()
df['AQI'] = df['AQI'].interpolate()
df.set_index('Date', inplace=True)

# Normalize AQI
scaler = MinMaxScaler()
aqi_scaled = scaler.fit_transform(df[['AQI']].values)

# Create sequences
def create_sequences(data, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

n_steps = 30
X, y = create_sequences(aqi_scaled, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Save model
model.save("lstm_aqi_model.h5")
print("✅ Model saved as 'lstm_aqi_model.h5'")
