# project2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from Crypto.Cipher import AES
import time
from sklearn.linear_model import LogisticRegression
def generate_noisy_data(x, noise_factor=0.2):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=x.shape)
    return x + noise
x_train = np.linspace(0, 2 * np.pi, 100)
y_train = np.sin(x_train)
x_train_noisy = generate_noisy_data(x_train)
y_train_noisy = generate_noisy_data(y_train)
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
model.fit(x_train_noisy, y_train_noisy, epochs=100, batch_size=10, validation_split=0.2)
x_test = np.linspace(0, 2 * np.pi, 100)
y_test = np.sin(x_test)
loss = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
def aes_encryption(data, key):
    # Initialize the AES cipher object with the key
    cipher = AES.new(key, AES.MODE_ECB)
    # Encrypt the data
    start_time = time.perf_counter()
    cipher_text = b""
    for i in range(0, len(data), 16):
        block = data[i:i+16]
        if len(block) < 16:
            block += b"\x00" * (16 - len(block))
        encrypted_block = cipher.encrypt(block)
        cipher_text += encrypted_block
    end_time = time.perf_counter()
    encryption_time = end_time - start_time
    return cipher_text, encryption_time
def perform_timing_analysis(iterations=100):
    data = b"Hello, world!" * 10
    key = b'Sixteen byte key'
    times_without_delay = []
    for _ in range(iterations):
        _, encryption_time = aes_encryption(data, key)
        times_without_delay.append(encryption_time)
    times_with_delay = []
    for _ in range(iterations):
        _, encryption_time = aes_encryption(data, key)
        time.sleep(0.01) 
        times_with_delay.append(encryption_time + 0.01)
    mean_without_delay = np.mean(times_without_delay)
    mean_with_delay = np.mean(times_with_delay)
    security_probability = 1 - (mean_without_delay / mean_with_delay)
    security_percentage = max(0, security_probability) * 100
    print("Average Encryption Time without Delay:", mean_without_delay, "seconds")
    print("Average Encryption Time with Delay:", mean_with_delay, "seconds")
    print("Security Probability against Timing Attacks:", security_percentage, "%")
def conjanct_analysis(data, key, iterations=100):
    conjanct_model = LogisticRegression()
    features = []
    labels = []
    for _ in range(iterations):
        _, encryption_time = aes_encryption(data, key)
        features.append(encryption_time)
        labels.append(random.choice([0, 1]))  
    features = np.array(features).reshape(-1, 1)
    labels = np.array(labels)
    conjanct_model.fit(features, labels)
    accuracy = conjanct_model.score(features, labels)
    print("Conjanct Analysis Accuracy:", accuracy * 100, "%")

if __name__ == "__main__":
    perform_timing_analysis()
    print("مدل یادگیری مقاوم آموزش داده شد و ارزیابی شد")    data = b"Hello, world!" * 10  پ
    conjanct_analysis(data, key)
