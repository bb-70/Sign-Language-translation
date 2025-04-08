import cv2
import numpy as np
import mediapipe as mp
import os

# ==============================
# 설정
# ==============================
label = '슬그머니'  # 저장할 수어 이름
DATA_PATH = '/Users/soyeon/4-1/종합설계/translation/sign_data'
VIDEO_LIST = [
    '4-1/종합설계/translation/SLV/슬그머니1.mp4'
]

# 저장 폴더 생성
save_dir = os.path.join(DATA_PATH, label)
os.makedirs(save_dir, exist_ok=True)

# 기존에 저장된 npy 개수부터 시작
existing = os.listdir(save_dir)
saved_count = len([f for f in existing if f.endswith('.npy')])

# ==============================
# MediaPipe 초기화
# ==============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# ==============================
# 영상 목록 처리
# ==============================
for video_path in VIDEO_LIST:
    cap = cv2.VideoCapture(video_path)
    sequence = []

    print(f"[INFO] 영상 처리 시작: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print("⚠️ 이미지 shape 오류 → 건너뜀")
            continue

        image.flags.writeable = False
        pose_result = pose.process(image)
        hands_result = hands.process(image)

        keypoints = []

        # 포즈
        if pose_result.pose_landmarks:
            for idx in [11, 13, 15, 12, 14, 16]:
                lm = pose_result.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0]*18)

        # 손
        if hands_result.multi_hand_landmarks:
            detected_hands = hands_result.multi_hand_landmarks[:2]
            for hand in detected_hands:
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if len(detected_hands) == 1:
                keypoints.extend([0]*63)
        else:
            keypoints.extend([0]*126)

        if len(keypoints) != 144:
            print(f"❗ 키포인트 길이 오류: {len(keypoints)}, 건너뜀")
            continue

        sequence.append(keypoints)

        if len(sequence) == 30:
            npy_path = os.path.join(save_dir, f'{saved_count}.npy')
            np.save(npy_path, np.array(sequence))
            print(f"✅ {saved_count}.npy 저장 완료")
            sequence = []
            saved_count += 1

    cap.release()

print(f"[🎉 완료] 총 저장된 시퀀스 수: {saved_count}")
cv2.destroyAllWindows()
