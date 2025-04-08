import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib  # encoder 저장용

# 데이터 경로
DATA_PATH = 'sign_data'
labels = [l for l in os.listdir(DATA_PATH) if not l.startswith('.')]
sequences, sequence_labels = [], []

for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    for file in os.listdir(label_path):
        if file.startswith('.'):
            continue
        sequence = np.load(os.path.join(label_path, file))
        if sequence.shape == (30, 144):
            sequences.append(sequence)
            sequence_labels.append(label)


# 전처리
X = np.array(sequences)
le = LabelEncoder()
y = le.fit_transform(sequence_labels)
y_cat = to_categorical(y)

# 저장 (모델 외에도 encoder도!)
np.save('X.npy', X)
np.save('y.npy', y_cat)
joblib.dump(le, 'label_encoder.pkl')

# 학습/검증 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# LSTM 모델
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(30, 144)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(sequence_labels)), activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# 모델 저장
model.save('sign_model.h5')


import os

