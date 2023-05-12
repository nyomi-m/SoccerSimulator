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

# Ball tracking code based off program by Adrian Rosebrock from
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/ (found by Andrew)

# Setup: allow parsing of arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())
#"C:\Program Files\Python39\python.exe" C:/Users/leeba/PycharmProjects/SoccerSimulator/FunCode.py
# Initialize Position: Define camera calibration
print('\n\nWhich camera calibration should be used?\n')
print('Type "1" for new GoPro\n')
print('Type "2" for old GoPro\n')
print('Type "3" to add new calibration\n')
cameraType = input()
K = [[]]
if cameraType == '1':
    K = [[2936.0, 0, 2072.9],
         [0, 2865.5, 1591.4],
         [0, 0, 1]]
elif cameraType == '2':
    K = [[1713.2, 0, 2072.9],
         [0, 1721.2, 1591.4],
         [0, 0, 1]]
elif cameraType == '3':
    print('\nWhat is the x-focal length of the camera?\n')
    K = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
    K[0][0] = input()
    print('\nWhat is the y-focal length of the camera?\n')
    K[1][1] = input()
    print('\nWhat is the x-shift of the camera?\n')
    K[0][2] = input()
    print('\nWhat is the y-shift of the camera?\n')
    K[1][2] = input()

fx = K[0][0]       # in pixels
fy = K[1][1]       # in pixels
x_pos = []         # in inches
y_pos = []         # in inches
z_pos = []         # in inches
ball_diameter = 9  # standard soccer ball diameter in inches

# Initialize Color: define a range of HSV color of the ball to distinguish it from the surroundings
# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)
# greenLower = (0 , 0 , 0)
# greenUpper = (255, 255 ,255)
# greenMiddle = (34,70,84)

greenLower = (28, 90, 70)
greenUpper = (36, 255, 255)
yellowBrownLower = (26, 150, 40)
yellowBrownUpper = (34, 240, 100)
brownLower = (26, 150, 30)
brownUpper = (30, 255, 80)

pts = deque(maxlen=args["buffer"])

# Setup: when we don't pull a video from commandline, open file explorer to find video
# Nyomi Found this solution from https://stackoverflow.com/questions/54307228/how-to-show-videos-frame-by-frame
# -with-key-presses-with-python-and-opencv

if not args.get("video", False):
    root = Tk()
    root.withdraw()
    root.focus_force()
    root.wm_attributes('-topmost', 1)
    
    filename = askopenfilename(parent=root)
    # filename = "C:\\Users\\kosikoaj\\Documents\\Rose\\Junior\\JuniorSpring\\CSSE461\\GoProVideos\\GX010024.MP4"
    vs = cv2.VideoCapture(filename)
else:

    vs = cv2.VideoCapture(args["video"])

# Pause to let video player boot up
time.sleep(2.0)

# Main Program Loop:
radius = 1
center = [0, 0]
frames_hidden = 0
while True:
    # Grab current frame:
    frame = vs.read()
    frame = frame[1]
    if frame is None:
        break

    # Prep frame: resize, convert to hsv, and apply Gaussian Blur
    dimensions = frame.shape
    frame = imutils.resize(frame, width=600)
    resized_dimensions = frame.shape
    height = resized_dimensions[0]
    width = resized_dimensions[1]
    x_scale = width/dimensions[1]
    y_scale = height/dimensions[0]

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Mask frame: blacks out all colors that fall outside the specified range and cleans up outliers
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask2 = cv2.inRange(hsv, yellowBrownLower, yellowBrownUpper)
    mask3 = cv2.inRange(hsv, brownLower, brownUpper)
    mask = mask | mask2 | mask3
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    # Find contours: get all edges and center of ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    radius = 1
    if len(cnts) > 0:
        # Assume largest contour is edge of ball, use to compute center and size
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # **Only registers a ball if the contour is big enough**
        if radius > 2:
            # Draws circle in frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # Adds new center location to list of previous centers
    pts.appendleft(center)

    # Compute Position:
    # (average the depth based on x and y focal lengths for increased accuracy)
    if len(z_pos) > 1 or radius > 2:
        if radius < 2:
            frames_hidden = frames_hidden + 1
        else:
            if frames_hidden > 0:
                old_z = z_pos[len(z_pos)-1]
                old_x = x_pos[len(x_pos) - 1]
                old_y = y_pos[len(y_pos) - 1]
                new_z = 0.5 * (((ball_diameter * fx * x_scale) / (2 * radius)) + ((ball_diameter * fy * y_scale) / (2 * radius)))
                new_x = (new_z * (center[0] - (width / 2))) / (fx * x_scale)
                new_y = (new_z * (center[1] - (height / 2))) / (fy * y_scale)
                dz = (new_z - old_z) / (frames_hidden + 1)
                dx = (new_x - old_x) / (frames_hidden + 1)
                dy = (new_y - old_y) / (frames_hidden + 1)
                for i in range(frames_hidden):
                    z_pos.append(z_pos[len(z_pos)-1] + dz)
                    x_pos.append(x_pos[len(x_pos) - 1] + dx)
                    y_pos.append(y_pos[len(y_pos) - 1] + dy)
            else:
                z_pos.append(0.5 * (((ball_diameter * fx * x_scale) / (2 * radius)) + ((ball_diameter * fy * y_scale) / (2 * radius))))
                x_pos.append((z_pos[len(z_pos)-1] * (center[0] - (width / 2))) / (fx * x_scale))
                y_pos.append((z_pos[len(z_pos)-1] * (center[1] - (height / 2))) / (fy * y_scale))

            print(x_pos[len(x_pos)-1], ' ',y_pos[len(y_pos)-1], ' ',z_pos[len(z_pos)-1], ' ')


    # Draw dot at center: draws center and trail of previous center locations
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # Tapers line between previous centers
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Display video frame:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Stop key:
    if key == ord("q"):
        break


# Stop stream when done:
vs.release()

# Close program:
cv2.destroyAllWindows()

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = z_pos
xline = x_pos
yline = y_pos
# ax.plot3D(xline, yline, zline, 'gray')
# plt.show()
# while True:
#     True


# def gen(n):
#     phi = 0
#     while phi < 2*np.pi:
#         yield np.array([np.cos(phi), np.sin(phi), phi])
#         phi += 2*np.pi/n

# def update(num, data, line):
def update(num, x_pos,y_pos,z_pos, line):
    # line.set_data(data[:2, :num])
    line.set_data(x_pos[:num], y_pos[:num])
    line.set_3d_properties(z_pos[:num])
    # line.set_3d_properties(data[2, :num])

# N = 100
# N = len(x_pos)
N = len(x_pos)
# data = np.array(list(gen(N))).T
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
line, = ax.plot(x_pos[0:1], y_pos[0:1], z_pos[0:1])

# Setting the axes properties
ax.set_xlim3d([min(x_pos), max(x_pos)])
ax.set_xlabel('X')

ax.set_ylim3d([min(y_pos), max(y_pos)])
ax.set_ylabel('Y')

ax.set_zlim3d([min(z_pos), max(z_pos)])
ax.set_zlabel('Z')

anim = animation.FuncAnimation(fig, update, N, fargs=(x_pos,y_pos,z_pos, line), interval=20, blit=False)
plt.show()