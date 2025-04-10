import os
import cv2
import numpy as np
import mediapipe as mp

# ==============================
# 경로 설정
# ==============================
VIDEO_ROOT = '/Users/soyeon/4-1/종합설계/translation/SLV'
SAVE_ROOT = '/Users/soyeon/4-1/종합설계/translation/sign_data'

# ==============================
# MediaPipe 초기화
# ==============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# ==============================
# 라벨 폴더 순회
# ==============================
for label in os.listdir(VIDEO_ROOT):
    label_path = os.path.join(VIDEO_ROOT, label)
    if not os.path.isdir(label_path):
        continue

    print(f"\n[⚙️ 처리 시작] 라벨: {label}")
    save_dir = os.path.join(SAVE_ROOT, label)
    os.makedirs(save_dir, exist_ok=True)

    existing = os.listdir(save_dir)
    saved_count = len([f for f in existing if f.endswith('.npy')])

    # 모든 영상 파일 탐색
    video_files = [f for f in os.listdir(label_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(label_path, video_file)
        cap = cv2.VideoCapture(video_path)
        sequence = []

        print(f"🎥 영상 처리 중: {video_file}")

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

            # 포즈: 어깨~손목 (6개 x 3)
            if pose_result.pose_landmarks:
                for idx in [11, 13, 15, 12, 14, 16]:
                    lm = pose_result.pose_landmarks.landmark[idx]
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0] * 18)

            # 손: 최대 양손
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

print("\n🎉 전체 처리 완료!")
cv2.destroyAllWindows()

sample = np.load('/Users/soyeon/4-1/종합설계/translation/sign_data/못생기다/0.npy')
print("Shape:", sample.shape)
print("Sample data (첫 프레임):", sample[0][:10])  # 앞 10개만

