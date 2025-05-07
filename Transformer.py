import os
import numpy as np
import joblib
import csv
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 데이터 로딩 (기존과 동일)
# ===============================
def load_data_flat(data_path):
    sequences, labels = [], []
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
# Transformer block
# ===============================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    x_skip = x
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Add()([x, x_skip])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ===============================
# 모델 구성
# ===============================
def build_transformer_model(input_shape, num_classes, head_size=64, num_heads=4, ff_dim=128, num_blocks=2, dropout=0.3):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ===============================
# 학습 및 평가
# ===============================
def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true_classes, y_pred_classes)
    prec = precision_score(y_true_classes, y_pred_classes, average='macro')
    rec = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    return acc, prec, rec, f1

# ===============================
# 메인
# ===============================
def main():
    DATA_PATH = '4-1/종합설계/translation/sign_data'
    MODEL_SAVE_DIR = '4-1/종합설계/translation/models'
    CSV_LOG_PATH = os.path.join(MODEL_SAVE_DIR, 'transformer_results.csv')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    with open(CSV_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Blocks', 'Heads', 'Dropout', 'Accuracy', 'Precision', 'Recall', 'F1'])

    X, y_labels = load_data_flat(DATA_PATH)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    y_cat = to_categorical(y)
    joblib.dump(le, os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    input_shape = (30, 144)
    num_classes = len(np.unique(y))
    block_options = [2, 3]
    head_options = [4, 8]
    dropout_options = [0.3, 0.5]

    best_f1 = 0
    best_model = None
    best_name = ""
    best_acc = best_prec = best_rec = 0

    for blocks in block_options:
        for heads in head_options:
            for dropout in dropout_options:
                print(f"\n🚀 Training: blocks={blocks}, heads={heads}, dropout={dropout}")
                model = build_transformer_model(input_shape, num_classes, num_heads=heads, num_blocks=blocks, dropout=dropout)
                acc, prec, rec, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, model)

                print(f"📊 Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
                model_name = f"transformer_model_b{blocks}_h{heads}_d{int(dropout*100)}.h5"
                model_path = os.path.join(MODEL_SAVE_DIR, model_name)
                model.save(model_path)

                with open(CSV_LOG_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_name, blocks, heads, dropout, acc, prec, rec, f1])

                if f1 > best_f1:
                    best_f1 = f1
                    best_acc = acc
                    best_prec = prec
                    best_rec = rec
                    best_model = model
                    best_name = model_name

    if best_model:
        best_model.save(os.path.join(MODEL_SAVE_DIR, 'transformer_model_best_f1.h5'))
        print("\n🏆 [베스트 모델 성능 요약]")
        print(f"📌 모델 이름     : {best_name}")
        print(f"📊 Accuracy      : {best_acc:.2f}")
        print(f"🎯 Precision     : {best_prec:.2f}")
        print(f"📈 Recall        : {best_rec:.2f}")
        print(f"⭐ F1 Score       : {best_f1:.2f}")
        print(f"\n💾 저장 경로     : transformer_model_best_f1.h5")
        print(f"📄 전체 결과 CSV : {CSV_LOG_PATH}")

if __name__ == '__main__':
    main()
