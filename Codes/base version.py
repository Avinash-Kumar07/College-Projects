import cv2 as cv
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from kalman import kalman

###################### Kalman initial ########################
# status that Kalman is updating. This is the initial value
degree = np.pi/180
a = np.array([0, 900])

fps = 60
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
# sus incertidumbres
P = np.diag([1000,1000,1000,1000])**2

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