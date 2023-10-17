# load state.json and results_kolo_zlute.json

import json
import numpy as np
import cv2
import pprint
import glob

scalingFactor = 1.6
landmarks_file = 'gt/kolo_cerne.json'
results_files = glob.glob("predictions/*/results_kolo_cerne.json")
landmark_names = ["foot", "heel", "ankle",
                  "knee", "hip", "shoulder", "elbow", "wrist"]
show = False
video_path = 'kolo_cerne.mp4'

keypoint_names = ["left_small_toe", "left_heel","left_ankle", "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]

def getKeypointMapping(keypoint_names, results):
    name_dict = results['meta_info']["keypoint_name2id"]
    mapping = []
    for name in keypoint_names:
        if name in name_dict:
            mapping.append(name_dict[name])
        else:
            mapping.append(-1)
    return mapping

def flipKeypoints(mapping, results):
    flip_indices = results['meta_info']['flip_indices']
    for i, index in enumerate(mapping):
        if index != -1:
            mapping[i] = flip_indices[index]
    return mapping

def distance(p1, p2):
    return np.sqrt((p1['x'] - p2[0])**2 + (p1['y'] - p2[1])**2)

def loadJSON(path):
    with open(path) as f:
        return json.load(f)

def getLandmarks(landmarksStore):
    return [[{"x": lm["x"]*scalingFactor, "y": lm["y"]*scalingFactor}
              for lm in landmarkList] for landmarkList in landmarksStore]

def isFacingLeft(landmarks):
    # check first frame predictions to see if left wrist is more to the left than left hip
    # left wrist
    wrist = landmarks[0][7]
    # left hip
    hip = landmarks[0][4]
    return wrist['x'] < hip['x']

def calcDistances(landmarks, predictions):
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
    return distances

def visualizeLandmarks(landmarks, predictions, video_path):
    cap = cv2.VideoCapture(video_path)
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
    


gt = loadJSON(landmarks_file)
landmarks = getLandmarks(gt['$landmarksStore'])

for results_file in results_files:
    model_name = results_file.split('\\')[1]
    print("Model: " + model_name)

    results = loadJSON(results_file)



    predictions = results['instance_info']
    assert len(predictions) == len(landmarks)

    keypoint_mapping = getKeypointMapping(keypoint_names, results)

    facing_left = isFacingLeft(landmarks)

    if not facing_left:
        keypoint_mapping = flipKeypoints(keypoint_mapping, results)


    distances = np.array(calcDistances(landmarks, predictions))
    landmarks_distances = np.mean(distances, axis=0)

    print("Landmarks distances: \n" + pprint.pformat(dict(zip(landmark_names, landmarks_distances))))
    print("Total avg distance: " + str(np.mean(landmarks_distances, where=landmarks_distances != -1)))

    print()
    if show:
        visualizeLandmarks(landmarks, predictions, video_path)
