import cv2
import numpy as np
import mediapipe as mp
import os

label = '어지럽다'
DATA_PATH = 'sign_data'
os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture('/Users/soyeon/4-1/종합설계/SLV/어지럽다.mp4')  # 웹캠 사용. 또는 '배부르다.mp4'

sequence = []
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    pose_result = pose.process(image)
    hands_result = hands.process(image)

    keypoints = []

    # 포즈: 어깨, 팔꿈치, 손목 (총 6개 x 3 = 18개)
    if pose_result.pose_landmarks:
        for idx in [11, 13, 15, 12, 14, 16]:
            lm = pose_result.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*18)

    # 핸드: 최대 2손 x 21개 x 3 = 126개
    if hands_result.multi_hand_landmarks:
        for hand in hands_result.multi_hand_landmarks:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        # 만약 한 손만 인식되면, 다른 손은 63개 0으로 패딩
        if len(hands_result.multi_hand_landmarks) == 1:
            keypoints.extend([0]*63)
    else:
        keypoints.extend([0]*126)

    # 길이 확인
    if len(keypoints) != 144:
        print(f"경고: keypoints 길이 = {len(keypoints)}, 건너뜀")
        continue

    sequence.append(keypoints)

    if len(sequence) == 30:
        try:
            np_sequence = np.array(sequence)
            np.save(os.path.join(DATA_PATH, label, f'{saved_count}.npy'), np_sequence)
            print(f'{saved_count}.npy 저장 완료')
            saved_count += 1
            sequence = []
        except Exception as e:
            print(f"저장 중 오류 발생: {e}")
            sequence = []

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
