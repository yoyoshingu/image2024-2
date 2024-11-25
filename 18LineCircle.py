# 2024.11.25 이미지처리수업
# Canny edge detector를 이용한 라인과 원의 검출

import cv2
import numpy as np
from cv2 import COLOR_BGR2GRAY

# 1. 허프변환을 이용한 직선검출 cv2.HoughLines()

img = cv2.imread('./img/paper.jpg')
# 컬러이미지는 흑백변환
#img = cv2.imread("./img/children.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
print(lines)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 직선 방정식으로부터 직선의 시작점과 끝점 계산
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # 직선 그리기
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. 확률적 허프변환을 이용한 직선검출 cv2.HoughLinesP()


img = cv2.imread('./img/paper.jpg')
edges = cv2.Canny(img, 50, 150, apertureSize=3)
linesp = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=20)

print(linesp)
if linesp is not None:
    for line in linesp:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3 허프변환을 이용한 원 검출 cv2.HoughCircles()

img = cv2.imread('./img/coins_connected.jpg')
imgray = cv2.cvtColor(img, COLOR_BGR2GRAY)
circles = cv2.HoughCircles(imgray, method=cv2.HOUGH_GRADIENT,
                           dp=1, minDist=50, param2=120,
                           minRadius=10, maxRadius=100)
circles = np.int32(circles)
for circle in circles[0, :]:
    cx, cy, r = circle
    cv2.circle(img, (cx, cy), r, (0,0,255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
