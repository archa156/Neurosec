import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('C:\\Users\\REDMI\\Music\\PROJECTS\\ganlstm\\KDDTrain+.txt.zip')

# Select numeric columns for normalization
numeric_columns = data.select_dtypes(include=[np.number]).columns
values = data[numeric_columns].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# Save the scaler for real-time anomaly detection
joblib.dump(scaler, 'scaler.pkl')

# Separate normal and anomalous data
normal_data = scaled_values[data['normal'] == 'normal']
anomalous_data = scaled_values[data['normal'] != 'normal']

print("✅ Data preprocessing complete.")

# 🔹 Build GAN Generator
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='tanh')  # Ensure output shape matches real data
    ])
    return model

# 🔹 Build GAN Discriminator
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

latent_dim = 10
generator = build_generator(latent_dim, normal_data.shape[1])  # Ensure generator output is 39-dimensional
discriminator = build_discriminator(normal_data.shape[1])

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
gan_input = tf.keras.layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

print("✅ GAN model setup complete.")

# 🔹 Build LSTM Model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

time_step = 10
lstm_model = build_lstm_model((time_step, normal_data.shape[1]))

print("✅ LSTM model setup complete.")

# Save models
generator.save('gan_generator.h5')
lstm_model.save('lstm_model.h5')
print("✅ Models saved successfully.")

# Save preprocessed data
preprocessed_data = pd.DataFrame(scaled_values, columns=numeric_columns)
preprocessed_data.to_csv('scaled_data.csv', index=False)
print("✅ Preprocessed data saved.")
