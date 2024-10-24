import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Read data from CSV
df = pd.read_csv('power_data.csv')
df['timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df['power_normalized'] = scaler.fit_transform(df['Units'].values.reshape(-1, 1))

# Create sequences for LSTM training
sequence_length = 5  # Adjust as needed
X, y = [], []

for i in range(len(df) - sequence_length):
    X.append(df['power_normalized'].iloc[i:i + sequence_length].values)
    y.append(df['power_normalized'].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_val, y_val))

# Make predictions
y_pred = model.predict(X_val)

# Inverse transform the scaled data to get actual values
y_pred = scaler.inverse_transform(y_pred)
y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_val):], y_val, label='Actual Consumption', color='blue')
plt.plot(df.index[-len(y_pred):], y_pred, label='Predicted Consumption', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Consumption')
plt.legend()
plt.title('Actual vs. Predicted Consumption')
plt.grid(True)

# Save the plot
plt.savefig('img.png')

plt.show()
