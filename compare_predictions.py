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
show = True
video_path = 'kolo_cerne.mp4'

keypoint_names = ["left_small_toe", "left_heel", "left_ankle",
                  "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]


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


def landmarksToArr(landmarks):
    arr = []
    for landmark in landmarks:
        frame_arr = []
        for point in landmark:
            point_arr = []
            point_arr.append(point['x'])
            point_arr.append(point['y'])
            frame_arr.append(point_arr)
        arr.append(frame_arr)
    arr = np.array(arr)
    assert arr.shape == (len(landmarks), 8, 2), "shape is " + str(arr.shape)
    return arr


def predictionsToArr(predictions, keypoint_mapping):
    arr = []
    for prediction in predictions:
        frame_arr = []
        for i in range(8):
            if keypoint_mapping[i] != -1:
                point_arr = []
                point_arr.append(
                    prediction['instances'][0]['keypoints'][keypoint_mapping[i]][0])
                point_arr.append(
                    prediction['instances'][0]['keypoints'][keypoint_mapping[i]][1])
                frame_arr.append(point_arr)
            else:
                frame_arr.append([-1, -1])
        arr.append(frame_arr)
    arr = np.array(arr)
    assert arr.shape == (len(predictions), 8, 2), "shape is " + str(arr.shape)
    return arr


def calcDistancesArr(landmarks_arr, predictions_arr):
    assert landmarks_arr.shape == predictions_arr.shape, "shapes are " + \
        str(landmarks_arr.shape) + " and " + str(predictions_arr.shape)
    dists = []
    for (landmark, prediction) in zip(landmarks_arr, predictions_arr):
        max_x = max(landmark, key=lambda p: p[0])[0]
        min_x = min(landmark, key=lambda p: p[0])[0]
        max_y = max(landmark, key=lambda p: p[1])[1]
        min_y = min(landmark, key=lambda p: p[1])[1]
        normalization = np.sqrt((max_x - min_x) * (max_y - min_y))
        frame_dists = []
        for (landmark_point, prediction_point) in zip(landmark, prediction):
            if landmark_point[0] != -1 and prediction_point[0] != -1:
                dist = np.sqrt(
                    np.sum((landmark_point-prediction_point)**2)) / normalization * 100
            else:
                dist = -1
            frame_dists.append(dist)
        dists.append(frame_dists)
    dists = np.array(dists)
    print(dists.shape)
    return dists


def visualizeLandmarks(landmarks_arr, predictions_arr, video_path):
    cap = cv2.VideoCapture(video_path)
    # read frames
    frame_idx = 0
    ret, frame = cap.read()
    while ret:
        # draw landmarks

        landmark = landmarks_arr[frame_idx]
        for point in landmark:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 5, (255, 0, 0), -1)

        # draw predictions
        prediction = predictions_arr[frame_idx]
        for point in prediction:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 5, (255, 255, 0), -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

        ret, frame = cap.read()
        frame_idx += 1


def main():

    gt = loadJSON(landmarks_file)
    landmarks = getLandmarks(gt['$landmarksStore'])
    facing_left = isFacingLeft(landmarks)

    landmarks = landmarksToArr(landmarks)

    for results_file in results_files:
        model_name = results_file.split('\\')[1]
        print("Model: " + model_name)

        results = loadJSON(results_file)
        keypoint_mapping = getKeypointMapping(keypoint_names, results)

        predictions = results['instance_info']
        assert len(predictions) == len(landmarks)
        predictions = predictionsToArr(predictions, keypoint_mapping)

        if not facing_left:
            keypoint_mapping = flipKeypoints(keypoint_mapping, results)

        distances = calcDistancesArr(landmarks, predictions)

        landmarks_distances = np.mean(distances, axis=0)

        print("Landmarks distances: \n" +
              pprint.pformat(dict(zip(landmark_names, landmarks_distances))))
        print("Total avg distance: " +
              str(np.mean(landmarks_distances, where=landmarks_distances != -1)))

        print()
        if show:
            visualizeLandmarks(landmarks, predictions, video_path)


if __name__ == "__main__":
    main()
