import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# بارگذاری داده ها
data = pd.read_csv("data.csv")

# جدا کردن داده ها
accelerometer_data = data["accelerometer"]
heart_rate_data = data["heart_rate"]
location_data = data["location"]
labels = data["label"]

# تبدیل داده ها به توالی
sequences = []
for i in range(len(data)):
  sequence = [accelerometer_data[i], heart_rate_data[i], location_data[i]]
  sequences.append(sequence)

# تعریف مدل 
model = Sequential()
model.add(LSTM(128, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(6, activation="softmax"))

# کامپایل مدل
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# آموزش مدل
model.fit(sequences, labels, epochs=100)

# پیش‌بینی وضعیت
new_sequence = [[1.2, 70, (40.7127, -74.0059)]]
prediction = model.predict(new_sequence)

# نمایش نتیجه
print(f"وضعیت پیش‌بینی شده: {labels[np.argmax(prediction)]}")
