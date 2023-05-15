from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d
from tkinter import *
import math 

import matplotlib

def ballsToTheWallsSpeed(xVec, yVec, zVec, frameRate):
    
    #framerate that we're assuming, please comment out if we have something that's actually passed in
    frameRate = 24
    
    xMin = min(xVec)
    xMax = max(xVec)
    xFirst = xVec[0]
    minDiff = abs(xFirst-xMin)
    maxDiff = abs(xFirst-xMax)
    startInd = 0
    

    goingLeft = True
    if(minDiff<maxDiff):
        goingLeft = False

    wallInd = 0
    if(goingLeft):
        for x in range(len(xVec)):
            if(xVec[x]==xMin):
                wallInd = x
                break
    else:
        for x in range(len(xVec)):
            if(xVec[x]==xMax):
                wallInd = x
                break    

    for x in range(len(xVec)-1):
        if math.sqrt((xVec[x+1]-xVec[x])**2 + (yVec[x+1]-yVec[x])**2) > 100:
            startInd = x
            break
    

    print(xVec[wallInd])
    print(xVec[startInd])
    print(yVec[wallInd])
    print(yVec[startInd])
    print(zVec[wallInd])
    print(zVec[startInd])
    distance = ((xVec[startInd]-xVec[wallInd])**2 + (yVec[startInd]-yVec[wallInd])**2 + (zVec[startInd]-zVec[wallInd])**2)**0.5
    presqrt = (xVec[startInd]-xVec[wallInd])**2 + (yVec[startInd]-yVec[wallInd])**2 + (zVec[startInd]-zVec[wallInd])**2
    # interesting = (-56-475)**2 + (-14-76)**2 + (95-zVec[wallInd])**2)
    print("What should be inside of the sqrt "+str(presqrt))
    frameDiff = wallInd - startInd
    seconds = frameDiff/frameRate
    speed = distance/seconds
    print("Calculated distance: "+str(distance))
    print(frameDiff)
    print(seconds)
    print(speed)
    

