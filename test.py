import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import Counter

# ============================
# 모델 호출 & 인코더 로드
# ============================
model = load_model('4-1/종합설계/translation/models/transformer_model_best_f1.h5')
encoder = joblib.load('4-1/종합설계/translation/label_encoder.pkl')

# ============================
# MediaPipe 초기화
# ============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# ============================
# 테스트할 수어 영상 경로 (수정 실행)
# ============================
cap = cv2.VideoCapture('/Users/soyeon/4-1/종합설계/translation/SLV/배부르다/배부르다6.mp4')

sequence = []
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    pose_result = pose.process(image)
    hands_result = hands.process(image)

    keypoints = []

    # 포즈: 어깨, 팔꿈치, 손목
    if pose_result.pose_landmarks:
        for idx in [11, 13, 15, 12, 14, 16]:
            lm = pose_result.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*18)

    # 손: 양손 최대 2개
    if hands_result.multi_hand_landmarks:
        for hand in hands_result.multi_hand_landmarks:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        if len(hands_result.multi_hand_landmarks) == 1:
            keypoints.extend([0]*63)
    else:
        keypoints.extend([0]*126)

    # 누락 방지
    if len(keypoints) != 144:
        continue

    sequence.append(keypoints)

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)  # (1, 30, 144)
        prediction = model.predict(input_data)[0]
        confidence = np.max(prediction)
        pred_label = encoder.inverse_transform([np.argmax(prediction)])[0]

        # 신뢰도 높은 경우만 저장
        if confidence > 0.8:
            predictions.append(pred_label)
            print(f'🟢 예측 결과: {pred_label} ({confidence:.2f})')
        else:
            print(f'🔹 무시됨: {pred_label} ({confidence:.2f})')

        sequence = []

cap.release()
cv2.destroyAllWindows()

# ============================
# 최종 하나만 출력
# ============================
if predictions:
    final_word = Counter(predictions).most_common(1)[0][0]
    print(f'\n🎯 최종 번역 결과: {final_word}')
else:
    print("\n❗ 예측된 결과가 없습니다요. 죽어라이~!")


