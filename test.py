import os
import numpy as np
import joblib
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 데이터 로딩 (현재 구조 그대로!)
# ===============================
def load_data_flat(data_path):
    sequences = []
    labels = []

    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith('.npy'):
                sequence = np.load(os.path.join(label_path, file))
                if sequence.shape == (30, 144):
                    sequences.append(sequence)
                    labels.append(label)

    return np.array(sequences), labels

# ===============================
# 모델 생성
# ===============================
def build_model(input_shape, num_classes, num_units, num_layers, dropout_rate):
    model = Sequential()
    model.add(LSTM(num_units, return_sequences=(num_layers > 1), input_shape=input_shape))
    for i in range(1, num_layers):
        model.add(LSTM(num_units, return_sequences=(i < num_layers - 1)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(num_units // 2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# 학습 및 평가
# ===============================
def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_classes, y_pred_classes)
    prec = precision_score(y_true_classes, y_pred_classes, average='macro')
    rec = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    return acc, prec, rec, f1

# ===============================
# 메인 함수
# ===============================
def main():
    DATA_PATH = '4-1/종합설계/translation/sign_data'
    MODEL_SAVE_DIR = '4-1/종합설계/translation/models'
    CSV_LOG_PATH = os.path.join(MODEL_SAVE_DIR, 'model_results.csv')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # CSV 파일 초기화
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Units', 'Layers', 'Dropout', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # 데이터 로딩
    X, y_labels = load_data_flat(DATA_PATH)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    y_cat = to_categorical(y)

    joblib.dump(le, '4-1/종합설계/translation/label_encoder.pkl')
    print("✅ 라벨 인코더 저장 완료")

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    num_classes = len(np.unique(y))
    input_shape = (30, 144)

    unit_options = [64, 128]
    layer_options = [1, 2]
    dropout_options = [0.0, 0.25]

    best_f1 = 0
    best_model = None
    best_name = ""

    for units in unit_options:
        for layers in layer_options:
            for dropout in dropout_options:
                print(f"\n🚀 Training: units={units}, layers={layers}, dropout={dropout}")
                model = build_model(input_shape, num_classes, units, layers, dropout)
                acc, prec, rec, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, model)

                print(f"📊 Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

                model_name = f"sign_model_u{units}_l{layers}_d{int(dropout*100)}.h5"
                model_path = os.path.join(MODEL_SAVE_DIR, model_name)
                model.save(model_path)
                print(f"✅ 모델 저장 완료: {model_path}")

                # CSV 저장
                with open(CSV_LOG_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_name, units, layers, dropout, acc, prec, rec, f1])

                # F1 최고 성능 갱신
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_name = model_name

    if best_model:
        best_model.save(os.path.join(MODEL_SAVE_DIR, 'sign_model_best_f1.h5'))
        print(f"\n🏆 베스트 모델 저장 완료 (F1={best_f1:.2f}) → sign_model_best_f1.h5")
        print(f"📄 모든 결과는 {CSV_LOG_PATH} 에 저장됨")

if __name__ == '__main__':
    main()
