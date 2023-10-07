# load state.json and results_kolo_zlute.json

import json
import numpy as np
import cv2


def distance(p1, p2):
    return np.sqrt((p1['x'] - p2[0])**2 + (p1['y'] - p2[1])**2)


scalingFactor = 1.6

gt = {}
with open('state.json') as f:
    gt = json.load(f)

results = {}
with open('rtm-m-256x192/results_kolo_cerne.json') as f:
    results = json.load(f)


landmarks = gt['landmarksStore']
landmarks = [[{"x": lm["x"]*scalingFactor, "y": lm["y"]*scalingFactor}
              for lm in landmarkList] for landmarkList in gt['landmarksStore']]


predictions = results['instance_info']
assert len(predictions) == len(landmarks)


left_keypoint_mapping = [-1, -1, 15, 13, 11, 5, 7, 9]
right_keypoint_mapping = [-1, -1, 16, 14, 12, 6, 8, 10]
# check first frame predictions to see if left wrist is more to the left than left hip
# left wrist
wrist = predictions[0]['instances'][0]['keypoints'][9]
# left hip
hip = predictions[0]['instances'][0]['keypoints'][11]
facing_left = wrist[0] < hip[0]

print(wrist, hip)
keypoint_mapping = left_keypoint_mapping if facing_left else right_keypoint_mapping

# calculate distances between landmarks and predictions
distances = []
for i, landmark in enumerate(landmarks):
    frame_distances = []
    prediction = predictions[i]
    for j, point in enumerate(landmark):
        if keypoint_mapping[j] != -1:
            frame_distances.append(
                distance(point, prediction['instances'][0]['keypoints'][keypoint_mapping[j]]))
        else:
            frame_distances.append(-1)
    distances.append(frame_distances)

distances = np.array(distances)
landmarks_distances = np.mean(distances, axis=0)

print("Landmarks distances: " + str(landmarks_distances))
print("Total avg distance: " + str(np.mean(landmarks_distances)))

# open video and read it frame by frame
cap = cv2.VideoCapture('kolo_cerne.mp4')

# print(distances[0])

# read frames
frame_idx = 0
ret, frame = cap.read()
while ret:
    # draw landmarks
    landmark = landmarks[frame_idx]
    for point in landmark:
        cv2.circle(frame, (int(point['x']), int(
            point['y'])), 5, (255, 0, 0), -1)

    # draw predictions
    prediction = predictions[frame_idx]
    for i, point in enumerate(prediction['instances'][0]['keypoints']):
        if i in keypoint_mapping:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 5, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

    ret, frame = cap.read()
    frame_idx += 1
# print(landmarks)
