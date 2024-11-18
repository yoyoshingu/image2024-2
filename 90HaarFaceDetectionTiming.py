import cv2
import time
from cv2.data import haarcascades
img = cv2.imread('./img/graduate.jpg')

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

@timing
def haardetection():
    face_cascade = cv2.CascadeClassifier('./recdata/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./recdata/haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    #print(faces)

    for face in faces:
        fx, fy, fw, fh = face
        cv2.rectangle(img, (fx,fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        eyes = eye_cascade.detectMultiScale(gray[fy:fy+fh, fx:fx+fw])
        #print(eyes)
        for eye in eyes:
            ex, ey, ew, eh = eye
            cv2.rectangle(img, (ex + fx, ey + fy), (ex + fx + ew, ey + fy + eh), (0, 255, 0), 2)



haardetection()

import cv2
from retinaface import RetinaFace

img = cv2.imread('./img/graduate.jpg')
faces = RetinaFace.detect_faces(img)
# save pretrained model at
# C:\Users\<Your_Username>\.deepface\weights\


print(faces)
# results
# {'face_1': {'score': 0.9995705485343933, 'facial_area': [400, 95, 490, 207], 'landmarks':
# {'right_eye': [431.66342, 137.51898], 'left_eye': [473.41107, 138.92555],
# 'nose': [456.38644, 160.18672], 'mouth_right': [432.9685, 178.13237], 'mouth_left': [467.67404, 179.28166]}},
# 'face_2': {'score': 0.9995139837265015, 'facial_area': [155, 93, 251, 212],
# 'landmarks': {'right_eye': [182.71143, 143.53116],
# 'left_eye': [227.43083, 143.12755], 'nose': [206.50964, 167.45905],
# 'mouth_right': [190.28046, 189.27559], 'mouth_left': [220.71698, 189.10281]}}}


#You have tensorflow 2.17.0 and this requires tf-keras package. Please run `pip install tf-keras` or downgrade your tensorflow.

image = cv2.imread(img_path)
# Iterate over the detected faces
for key, face in faces.items():
    # Extract facial area coordinates
    facial_area = face["facial_area"]

    # Draw a bounding box around each face
    cv2.rectangle(image,
                  (facial_area[0], facial_area[1]),
                  (facial_area[2], facial_area[3]),
                  (0, 255, 0), 2)

    # Optionally, you can extract other face details like landmarks, confidence, etc.
    landmarks = face["landmarks"]
    for point in landmarks.values():
        point = tuple(map(int, point))
        cv2.circle(image, point, 2, (0, 0, 255), -1)

# Display the image with detected faces
cv2.imshow("Face Detection with RetinaFace", image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imshow("gray", img)
cv2.waitKey()
cv2.destroyAllWindows()

