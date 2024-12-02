# https://github.com/serengil/deepface
import cv2
from deepface import DeepFace
import numpy as np

# 이미지 파일 경로 설정
image_path = "./img/children.jpg"
image = cv2.imread(image_path)

# 백엔드	정확도	속도	특징	추천 사용 사례
# RetinaFace	매우 높음	보통	작은 얼굴 및 랜드마크 검출	정밀한 얼굴 검출이 필요한 경우
# MTCNN	높음	중간	얼굴 및 랜드마크 검출	일반적인 얼굴 검출
# OpenCV	중간	빠름	단순하고 빠름	실시간 검출, 간단한 환경
# Dlib	높음	빠름	정면 얼굴 검출에 강함	정면 얼굴이 주로 있는 환경
# SSD	보통	매우 빠름	실시간 객체 검출에 강함	속도가 중요한 실시간 환경

# 얼굴 검출 수행
# RetinaFace를 얼굴 검출 백엔드로 설정
#detections = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=False)

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'fastmtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

detections = DeepFace.extract_faces(img_path=image_path, detector_backend=backends[9], enforce_detection=False)



# 얼굴 검출 결과 확인 및 화면에 표시
if detections:
    for detection in detections:

        # 얼굴 좌표와 크기 추출
        facial_area = detection["facial_area"]
        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]

        # 검출된 얼굴에 사각형 표시
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = detection["face"] * 255
        face = face.astype(np.uint8)
        print(face)
        cv2.imshow(f"face{x}", face)

else:
    print("얼굴이 검출되지 않았습니다.")

# 결과 이미지 표시
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

