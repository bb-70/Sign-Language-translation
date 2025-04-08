import cv2
import mediapipe as mp
import numpy as np
import os

# ==============================
# 사용자 설정
# ==============================
video_path = '/Users/soyeon/4-1/종합설계/SLV/슬그머니.mp4'  # 🎥 수화 영상 경로
label = '슬그머니'  # 📌 수어 단어
save_dir = os.path.join('sign_data', label)
os.makedirs(save_dir, exist_ok=True)

# ==============================
# MediaPipe 초기화
# ==============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(video_path)

sequence = []
saved_count = 0

print(f"[INFO] 영상 시작: {video_path}")

while cap.isOpened():
    ret, frame = cap.read()

    # 📌 프레임 체크
    if not ret or frame is None:
        print("⚠️ 프레임 없음 — 영상 끝 또는 읽기 실패")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 📌 이미지 shape 체크
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print("⚠️ 이미지 shape 오류 → 건너뜀")
        continue

    image.flags.writeable = False

    # MediaPipe 추론
    pose_result = pose.process(image)
    hands_result = hands.process(image)

    keypoints = []

    # 포즈: 어깨~손목 (6개 x 3D = 18)
    if pose_result.pose_landmarks:
        for idx in [11, 13, 15, 12, 14, 16]:
            lm = pose_result.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 18)

    # 손: 최대 2손 (21점 x 3D = 63 per hand)
    if hands_result.multi_hand_landmarks:
        detected_hands = hands_result.multi_hand_landmarks[:2]  # ⚠️ 변수명 변경!
        for hand in detected_hands:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        if len(detected_hands) == 1:
            keypoints.extend([0]*63)  # 한 손만 감지 시 보정
    else:
        keypoints.extend([0]*126)  # 손 없음

    # 총 144개 좌표 확인
    if len(keypoints) != 144:
        print(f"❗ 키포인트 길이 오류: {len(keypoints)}, 건너뜀")
        continue

    sequence.append(keypoints)

    # 30프레임마다 시퀀스 저장
    if len(sequence) == 30:
        np.save(os.path.join(save_dir, f'{saved_count}.npy'), np.array(sequence))
        print(f"✅ {saved_count}.npy 저장 완료")
        sequence = []
        saved_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"[완료] 총 저장된 시퀀스 수: {saved_count}")
