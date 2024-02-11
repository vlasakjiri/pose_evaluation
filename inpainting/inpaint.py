import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

filename = 'kolo_cerne.mp4'
out_folder = "frames"
# read video file
cap = cv.VideoCapture(filename)

frame_idx = 0
ret, frame = cap.read()
# get number of frames
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# print progress bar
for _ in tqdm(range(num_frames)):
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

    # save inpainted frame set jpg quality to 80
    cv.imwrite(os.path.join(out_folder, f' {filename}_{frame_idx}.jpg'), dst, [
               int(cv.IMWRITE_JPEG_QUALITY), 80])
    # cv.imshow("mask", mask)
    # cv.imshow('dst', dst)
    # if cv.waitKey(10) & 0xFF == ord('q'):
    #     break
    ret, frame = cap.read()
    frame_idx += 1
