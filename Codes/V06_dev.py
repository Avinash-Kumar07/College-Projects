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
#---------------------------------------Creating Projectile-----------------------------------------#
proa = -2
prob = 10.4
proc = 0
px = np.linspace(0,7,30)
py = proa*px*px + prob*px + proc
plt.scatter(px,py)
#plt.show()
#==================================================================================================#
#------------------------------------------ Kalman initial ----------------------------------------#
# Status that Kalman will be updating. This is the initial value
degree = np.pi/180
a = np.array([0, -0.23])
pa = 2
fps = 28
#fps = 120 (choose according to cam)
dt = 1
t = np.arange(0,2.01,dt)
noise = 0.001

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
for i in range(len(px)):
    #---------------------------------------Kalman---------------------------------------------#
    xo = px[i]
    yo = py[i]

    mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
    m="normal"
    mm = True

    if(mm):
        listCenterX.append(xo)
        listCenterY.append(yo)
        listCenterY1.append(py[i])
        listCenterX1.append(px[i])
    listpoints.append((xo,yo,m))
    res += [(mu,P)]
	#------------------------------------- Prediction ------------------------------------------#
    mu2 = mu
    P2 = P
    res2 = []

    for _ in range(30):
        #time.sleep(0.01)
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


    ##for n in range(len(listCenterX)): # centre of roibox
        ##cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

		#for n in [-1]: #range(len(xe)): # xe e ye estimada
			#inccirclesize = (xu[n]*yu[n])
			#cv.circle(frame,(int(xe[n]),int(ye[n])),int(inccirclesize),(255, 255, 0),-1)

			#inccirclesize=(xu[n]+yu[n])/2
			#cv2.circle(frame,(int(xe[n]),int(ye[n])),int(inccirclesize),(255, 255, 0),1)

    for n in range(len(xp)): # x e y predicted
		#inccirclesize=(xpu[n]+ypu[n])/2
        ##cv2.circle(frame,(int(xp[n]),int(yp[n])),int(5),(0, 0, 255))
        xp1.append(xp[n])
        yp1.append(yp[n])

		#print("List of points\n")
		#for n in range(len(listpoints)):
			#print(listpoints[n])
        # Below if is used for bouncing of the ball
        #if(len(listCenterY)>4):
	#==========================================================================================
#-----------------------------------------Ploting Results-----------------------------------------#
ksize = 30  
xp2 = [xp1[i:i + ksize] for i in range(0, len(xp1), ksize)]
yp2 = [yp1[i:i + ksize] for i in range(0, len(yp1), ksize)]

for n in range(len(xp2)-2):
    #figure(n+1)
    plt.xlabel("Distance in X direction")
    plt.ylabel("Distance in Y direction")
    plt.scatter(xp2[n],yp2[n], color='yellow')
    #plt.plot(xp2[n],yp2[n],markersize=80,label = 'Predicted Trajectory')
    #plt.plot(listCenterX1,listCenterY1,'ko',label = 'Actual Trajectory')
    #plt.legend(loc=1,shadow = True, fontsize = "small")
    ax = plt.gca()
    #ax.set_ylim([0,15])
    #ax.set_xlim([0,8])
    plt.grid()
    
    #plt.savefig('/home/avinash/Study/Mini Project/Results/test03/'+str(n)+".png",dpi = 400)
plt.show()
#================================================================================================