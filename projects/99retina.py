from retinaface import RetinaFace
import cv2

imgfile = './img/smilings.jpg'
resp = RetinaFace.detect_faces(imgfile)
print(resp)

image = cv2.imread(imgfile)

for key, face in resp.items():
    farea = face['facial_area']
    cv2.rectangle(image, (farea[0], farea[1]), (farea[2], farea[3]), (0, 255,0), 2)
    cv2.putText(image, f"{face['score']:.3f}", (farea[0], farea[1]), cv2.FONT_HERSHEY_PLAIN,
  1,  (0,255,0), 1 )


cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#pip list --format=freeze > requirements.txt
#pip freeze > requirements.txt
#pip install -r requirements.txt