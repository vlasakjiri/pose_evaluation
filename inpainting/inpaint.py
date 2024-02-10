import numpy as np
import cv2 as cv

# read video file
cap = cv.VideoCapture('kolo_cerne.mp4')

ret, frame = cap.read()
while ret:
    # convert to CIELAB color space
    lab = cv.cvtColor(frame, cv.COLOR_RGB2LAB)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.inRange(lab, (0, 31, 137), (255, 100, 216), mask)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    inp_mask = cv.erode(mask, kernel, iterations=1)
    # dilate mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=1)

    dst = cv.inpaint(frame, mask, 15, cv.INPAINT_NS)

    # cv.imshow("mask", mask)
    cv.imshow('dst', dst)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
