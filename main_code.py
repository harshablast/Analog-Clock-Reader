import cv2
import numpy as np

img=cv2.imread('clock1.png',1)

gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img_binary=cv2.threshold(gray_image,127,255,0)
img_binary=cv2.bitwise_not(img_binary)

cv2.imshow('binary',img_binary)
edges=cv2.Canny(img_binary,10,200)
cv2.imshow('Edges',edges)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
circles = cv2.HoughCircles(im2,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
maxrad=0
centre=(0,0)
for i in circles[0,:]:
    if i[2]>maxrad:
        maxrad=i[2]
        centre=(i[0],i[1])

lines = cv2.HoughLinesP(img_binary,1,np.pi/360,200)
for i in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.circle(img,centre,maxrad,(0,255,0),2)
cv2.drawContours(img, contours, -1, (0,255,0), 1)
cv2.imshow('final',img)

cv2.waitKey(0)

