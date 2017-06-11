import cv2
import numpy

img=cv2.imread('clock2.png',0)

ret,img_binary=cv2.threshold(img,127,255,0)
img_binary=cv2.bitwise_not(img_binary)



contours=cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0]
#ctr=numpy.array(contours).reshape((-1,1,2)).astype(numpy.int32)
cv2.drawContours(img_binary,contours,-1, (0,255,0), 3)
cv2.imshow('Original',img)
cv2.imshow('binary',img_binary)
print(contours)


cv2.waitKey(0)

