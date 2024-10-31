import copy

import cv2

rate = 10
img = cv2.imread('./img/graduate.jpg')

x,y,w,h = cv2.selectROI('mosaic', img, False)
if w and h:
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (w//rate, h//rate) )
    roi = cv2.resize(roi, (w, h))
    img[y:y+h, x:x+w] = roi
    cv2.imshow('mosaic', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

while True:
    x,y,w,h = cv2.selectROI('mosaic', img, False)
    if w and h:
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (w//rate, h//rate) )
        roi = cv2.resize(roi, (w, h))
        img[y:y+h, x:x+w] = roi
        cv2.imshow('mosaic', img)
    else:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()