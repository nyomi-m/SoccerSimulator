import cv2

## Read
img = cv2.imread("Screenshot2.png")

## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
mask1 = cv2.inRange(hsv, (26,150, 30), (34, 240,100))

## mask o yellow (15,0,0) ~ (36, 255, 255)
mask2 = cv2.inRange(hsv, (28,90,65), (42, 200, 200))

mask3 = cv2.inRange(hsv, (26,150,30), (30, 255, 80))

## final mask and masked
mask = cv2.bitwise_or(mask1, mask2)
mask = cv2.bitwise_or(mask,mask3)
target = cv2.bitwise_and(img,img, mask=mask)

cv2.imwrite("target.png", target)

