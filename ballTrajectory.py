# importing libraries
import cv2
from color_detector import ColorDetector
import numpy as np

cd = ColorDetector()
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('OldProTake1.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        yellowBox = cd.detect(frame)
        print(yellowBox)
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()