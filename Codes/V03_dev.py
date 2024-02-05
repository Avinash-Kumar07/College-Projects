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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
#vs = cv2.VideoCapture("ball.mp4")

time.sleep(2.0)

###################### Kalman initial ########################
# status that Kalman is updating. This is the initial value
degree = np.pi/180
a = np.array([0, 3000])

fps = 30
#fps = 120
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


# x, y, vx, vy
mu = np.array([0,0,0,0])
# your uncertainties
P = np.diag([300,300,300,300])**2

#res = [(mu,P,mu)]
res=[]
N = 15 # to take an initial section and see what happens if the observation is later lost

sigmaM = 0.0001 # noise model
sigmaZ = 3*noise # It should be equal to the average noise of the imaging process. 10 pixels pje.

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
listpuntos=[]

xp1 = []
yp1 = []
xp2 = []
yp2 = []
#------------------------------------------------------------------------------------------------------

# keep looping
while True:

	bgimg = cv2.imread('whitebg.png')

	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	bgimg = imutils.resize(bgimg, width=600)
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
		# only proceed if the radius meets a minimum size
		#if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
		#cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
		
		#cv2.circle(frame, center, 5, (0, 0, 255), -1)

		#------------------------------Kalman---------------------------------------------#
		xo = x
		yo = y

		mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
		m="normal"
		mm = True

		if(mm):
			listCenterX.append(xo)
			listCenterY.append(yo)

		listpuntos.append((xo,yo,m))
		res += [(mu,P)]
		#-------------------------- Prediction -------------------------------------------#
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
			#cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

			cv2.circle(bgimg,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

		#for n in [-1]: #range(len(xe)): # xe e ye estimada
			#incertidumbre = (xu[n]*yu[n])
			#cv.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),-1)

			#incertidumbre=(xu[n]+yu[n])/2
			#cv2.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),1)

			#cv2.circle(bgimg,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),1)

		for n in range(len(xp)): # x e y predicha
			#incertidumbreP=(xpu[n]+ypu[n])/2
			#cv2.circle(frame,(int(xp[n]),int(yp[n])),int(5),(0, 0, 255))

			cv2.circle(bgimg,(int(xp[n]),int(yp[n])),int(5),(0, 0, 255))

		#print("Lista de puntos\n")
		#for n in range(len(listpuntos)):
			#print(listpuntos[n])

		if(len(listCenterY)>4):
			if((listCenterY[-5] < listCenterY[-4]) and(listCenterY[-4] < listCenterY[-3]) and (listCenterY[-3] > listCenterY[-2]) and (listCenterY[-2] > listCenterY[-1])):

				#print("REBOTE")
				listCenterY=[]
				listCenterX=[]
				listpuntos=[]
				res=[]
				mu = np.array([0,0,0,0])
				P = np.diag([100,100,100,100])**2

		#-----------------------------------------------------------------------------------------------		

		

	# update the points queue
	pts.appendleft(center)

	
	# show the frame to our screen
	#cv2.imshow("Frame", frame)

	cv2.imshow("bgimg", bgimg)

	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()

#----------------------------------------------------ALL SET----------------------------------------------------------------