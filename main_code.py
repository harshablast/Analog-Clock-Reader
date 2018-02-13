import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def get_angle(hand,centre):
    x_h=hand[0][0]
    y_h=hand[0][1]
    x_c=centre[0]
    y_c=centre[1]

    x_diff=x_h-x_c
    y_diff=y_h-y_c
    x_diff=float(x_diff)
    y_diff=float(y_diff)

    if(x_diff*y_diff>0):
        if(x_diff>=0 and y_diff>0):
            angle=np.pi-np.arctan(x_diff/y_diff)
        elif(x_diff<=0 and y_diff<0):
            angle=2*np.pi-np.arctan(x_diff/y_diff)
    elif(x_diff*y_diff<0):
        if(y_diff>=0 and x_diff<0):
            angle=(3*np.pi)/4+np.arctan(x_diff/y_diff)
        elif(y_diff<=0 and x_diff>0):
            angle=-np.arctan(x_diff/y_diff)

    return angle

def get_hour(hourangle):
    hour=hourangle//30
    if(hour==0):
        return 12
    else:
        return int(hour)

def get_minsec(angle):
    minsec=angle/(np.rad2deg(2*np.pi))*60
    return int(minsec)

img = cv2.imread('clock2.png', 1)
#img=cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
img_grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_inverse=cv2.bitwise_not(img_grayscale)

#thresholding part
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
thresh=cv2.erode(thresh,kernel,iterations=1)
edges=cv2.Canny(thresh,100,200)
cv2.imshow('EDGES',edges)
cv2.imshow('THRESHOLDED IMAGE',thresh)


#Circle Detection Part
circles = cv2.HoughCircles(img_inverse,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=200,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
maxrad=0
centre=()
for i in circles[0,:]:
    if i[2]>maxrad:
        maxrad=i[2]
        centre=(i[0],i[1])

#Displaying Circle Around Clock
cv2.circle(img,centre,maxrad,(0,255,0),2)
cv2.circle(img,centre,2,(0,0,255),3)

#Getting and Displaying Contours
im2,contours,hierarchy=cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img,contours, -1, (0, 255, 0), 1)


#Getting Contours which contain centre point within it
shortlist=[]
for i in contours:
    x, y, w, h = cv2.boundingRect(i)
    if (centre[0]<(x+w) and centre[0]>x) and (centre[1]>y and centre[1]<y+h):
        shortlist.append(i)
cv2.drawContours(img,shortlist,-1,(255,0,0),2)

#Getting Contour of Hands
minarea=0
hand_contour=[]
for i in shortlist:
    if minarea==0:
        minarea=cv2.contourArea(i)
        hand_contour=i
        continue
    if minarea>cv2.contourArea(i):
        minarea=cv2.contourArea(i)
        hand_contour=i

#cv2.drawContours(img,[hand_contour],-1,(255,0,0),2)

#Getting and Clustering Hull Points
hull = cv2.convexHull(hand_contour)
hull=sorted(hull, key=lambda x: x[0][0])

groups = [[hull[0][0]]]
for i in hull[1:]:
    if (abs(i[0][0]-groups[-1][-1][0])<=20) and (abs(i[0][1]-groups[-1][-1][1])<=20):
        groups[-1].append(i[0])
    else:
        groups.append([i[0]])

hull_points_clustered=[]

for i in groups:
    x_sum=0
    y_sum=0
    l=len(i)
    for j in range(l):
        x_sum+=i[j][0]
        y_sum+=i[j][1]
    x_ave=x_sum//l
    y_ave=y_sum//l
    hull_points_clustered.append([x_ave,y_ave])

#Getting Points of Hands
hand_points=[]
for i in hull_points_clustered:
    x=i[0]
    y=i[1]
    cv2.circle(img,(x,y), 2, (0, 0, 255), 3)
    if (x-centre[0])**2+(y-centre[1])**2 >= (maxrad/2)**2:
        hand_point_dist=((x-centre[0])**2+(y-centre[1])**2)
        hand_points.append([[x,y],int(hand_point_dist)])


#Assigning Length of each hand to the hands
for i in hand_points:
    x=i[0][0]
    y=i[0][1]
    cv2.circle(img,(x,y), 2, (255, 0, 255), 3)
    length_of_hand=str(i[1])
    cv2.putText(img,length_of_hand, (x,y), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

hand_points=sorted(hand_points, key=lambda x:x[1])

#Getting Time

hour_hand=hand_points[0]
minute_hand=hand_points[1]
second_hand=hand_points[2]

hour_angle=np.rad2deg(get_angle(hour_hand,centre))
minute_angle=np.rad2deg(get_angle(minute_hand,centre))
second_angle=np.rad2deg(get_angle(second_hand,centre))

hour=get_hour(hour_angle)
minute=get_minsec(minute_angle)
second=get_minsec(second_angle)

#Displaying Time
time_display="The time is: "+str(hour)+":"+str(minute)+":"+str(second)

cv2.putText(img,time_display, (centre[0]-150,centre[1]+50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('Final',img)

cv2.waitKey(0)

