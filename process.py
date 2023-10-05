# load state.json and results_kolo_zlute.json

import json
import os
import numpy as np
import cv2

scalingFactor = 1.6

gt = {}
with open('state.json') as f:
    gt = json.load(f)

results = {}
with open('rtm-m-256x192/results_kolo_cerne.json') as f:
    results = json.load(f)

print(gt.keys(), results.keys())

landmarks = gt['landmarksStore']
print(landmarks[0])
landmarks = [[{"x": lm["x"]*scalingFactor, "y": lm["y"]*scalingFactor}
              for lm in landmarkList] for landmarkList in gt['landmarksStore']]
print(len(landmarks))
print(landmarks[0])

predictions = results['instance_info']
print(len(predictions), len(landmarks))

# open video and read it frame by frame
cap = cv2.VideoCapture('kolo_cerne.mp4')


# read frames
frame_idx = 0
ret, frame = cap.read()
while ret:
    # draw landmarks
    landmark = landmarks[frame_idx]
    for point in landmark:
        cv2.circle(frame, (int(point['x']), int(
            point['y'])), 2, (0, 255, 0), -1)

    # draw predictions
    prediction = predictions[frame_idx]
    for point in prediction['instances'][0]['keypoints']:
        cv2.circle(frame, (int(point[0]), int(
            point[1])), 2, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

    ret, frame = cap.read()
    frame_idx += 1
# print(landmarks)
