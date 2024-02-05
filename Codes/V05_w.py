# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

from scipy.ndimage.filters import gaussian_filter
from umucv.kalman import kalman, ukf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot, figure, subplots

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
vs = cv2.VideoCapture(0)
time.sleep(2.0)
#------------------------------------------ Kalman initial ----------------------------------------#
# Status that Kalman will be updating. This is the initial value
degree = np.pi/180
a = np.array([0, 2000])
pa = 2700
fps = 28
#fps = 120 (choose according to cam)
dt = 1/fps
t = np.arange(0,2.01,dt)
noise = 3

F = np.array(
[1, 0, dt, 0,
0, 1, 0, dt,
0, 0, 1, 0,
0, 0, 0, 1 ]).reshape(4,4)

B = np.array(
[dt**2/2, 0,
0, dt**2/2,
dt, 0,
0, dt ]).reshape(4,2)

H = np.array(
[1,0,0,0,
0,1,0,0]).reshape(2,4)

mu = np.array([0,0,0,0]) # x, y, vx, vy
P = np.diag([pa,pa,pa,pa])**2 # your uncertainties

res=[] #res = [(mu,P,mu)]
N = 15 # to take an initial section and see what happens if the observation is later lost

sigmaM = 0.0001 # noise model
sigmaZ = 3*noise # It should be equal to the average noise of the imaging process. 10 pixels pje.

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
listpoints=[]
listCenterX1 =[]
listCenterY1 = []
xp1 = []
yp1 = []
xp2 = []
yp2 = []
#=================================================================================================
while True:
    _,frame = vs.read()
    if frame is None:
        print('No frame Detected')
        break
    frame = imutils.resize(frame, width=600)
    #frame = cv2.resize(frame,(800,600))
    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #print(center[0],center[1])
        #---------------------------------------Kalman---------------------------------------------#
        xo = center[0]
        yo = center[1]

        mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
        m="normal"
        mm = True

        if(mm):
            listCenterX.append(xo)
            listCenterY.append(yo)
            listCenterY1.append(600-center[1])
            listCenterX1.append(center[0])
        listpoints.append((xo,yo,m))
        res += [(mu,P)]
		#------------------------------------- Prediction ------------------------------------------#
        mu2 = mu
        P2 = P
        res2 = []

        for _ in range(30):
            mu2,P2,pred2= kalman(mu2,P2,F,Q,B,a,None,H,R)
            res2 += [(mu2,P2)]

        xe = [mu[0] for mu,_ in res]
        xu = [2*np.sqrt(P[0,0]) for _,P in res]
        ye = [mu[1] for mu,_ in res]
        yu = [2*np.sqrt(P[1,1]) for _,P in res]

        xp=[mu2[0] for mu2,_ in res2]
        yp=[mu2[1] for mu2,_ in res2]
        xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
        ypu = [2*np.sqrt(P[1,1]) for _,P in res2]


        for n in range(len(listCenterX)): # centre of roibox
            cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

		#for n in [-1]: #range(len(xe)): # xe e ye estimada
			#inccirclesize = (xu[n]*yu[n])
			#cv.circle(frame,(int(xe[n]),int(ye[n])),int(inccirclesize),(255, 255, 0),-1)

			#inccirclesize=(xu[n]+yu[n])/2
			#cv2.circle(frame,(int(xe[n]),int(ye[n])),int(inccirclesize),(255, 255, 0),1)

        for n in range(len(xp)): # x e y predicted
			#inccirclesize=(xpu[n]+ypu[n])/2
            cv2.circle(frame,(int(xp[n]),int(yp[n])),int(5),(0, 0, 255))
            xp1.append(xp[n])
            yp1.append(600-yp[n])

		#print("List of points\n")
		#for n in range(len(listpoints)):
			#print(listpoints[n])
        # Below if is used for bouncing of the ball
        #if(len(listCenterY)>4):
            #if((listCenterY[-5] < listCenterY[-4]) and(listCenterY[-4] < listCenterY[-3]) and (listCenterY[-3] > listCenterY[-2]) and (listCenterY[-2] > listCenterY[-1])):
        if(len(listCenterY)>2):
            if((listCenterY[-3] < listCenterY[-2]) and (listCenterY[-2] > listCenterY[-1])):  
                #print("REBOUND")
                listCenterY=[]
                listCenterX=[]
                listpoints=[]
                res=[]
                mu = np.array([0,0,0,0])
                P = np.diag([pa,pa,pa,pa])**2
		#==========================================================================================
    key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    cv2.imshow("Frame", frame) # show the frame to our screen
    #cv2.imshow("Mask", mask) # show the mask to our screen

vs.release() # release the camera
cv2.destroyAllWindows() # close all windows
#-----------------------------------------Ploting Results-----------------------------------------#
ksize = 30  
xp2 = [xp1[i:i + ksize] for i in range(0, len(xp1), ksize)]
yp2 = [yp1[i:i + ksize] for i in range(0, len(yp1), ksize)]

for n in range(len(xp2)-2):
    figure(n+1)
    plt.xlabel("Distance in X direction")
    plt.ylabel("Distance in Y direction")
    plt.plot(xp2[n],yp2[n],markersize=80,label = 'Predicted Trajectory')
    plt.plot(listCenterX1,listCenterY1,'ko',label = 'Actual Trajectory')
    plt.legend(loc=1,shadow = True, fontsize = "small")
    ax = plt.gca()
    ax.set_ylim([0,600])
    ax.set_xlim([0,600])
    plt.grid()
    plt.savefig('/home/avinash/Study/Mini Project/Results/test03/'+str(n)+".png",dpi = 400)
#================================================================================================