import cv2
from deepface import DeepFace

img_file = "./img/children.jpg"
image = cv2.imread(img_file)

actions = ['age', 'gender', 'race', 'emotion']
ar = DeepFace.analyze(img_file, actions=actions)
print(ar)
cv2.rectangle(image, (144,  94), (144 + 122, 94+122),(255,0,0), 2 )
cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#face verification
obj = DeepFace.verify(
  img1_path = "img1.jpg",
  img2_path = "img2.jpg",
  detector_backend = backends[0],
  align = alignment_modes[0],
)

#face recognition
dfs = DeepFace.find(
  img_path = "img.jpg",
  db_path = "my_db",
  detector_backend = backends[1],
  align = alignment_modes[0],
)

#embeddings
embedding_objs = DeepFace.represent(
  img_path = "img.jpg",
  detector_backend = backends[2],
  align = alignment_modes[0],
)

#facial analysis
demographies = DeepFace.analyze(
  img_path = "img4.jpg",
  detector_backend = backends[3],
  align = alignment_modes[0],
)

#face detection and alignment
face_objs = DeepFace.extract_faces(
  img_path = "img.jpg",
  detector_backend = backends[4],
  align = alignment_modes[0],
)