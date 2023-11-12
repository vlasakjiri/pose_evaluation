import numpy as np
import cv2
import json


def landmarksToArr(landmarks):
    arr = []
    for landmark in landmarks:
        frame_arr = []
        for point in landmark:
            point_arr = []
            if point == {}:  # empty landmark
                point_arr.append(-1)
                point_arr.append(-1)
            else:
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
        normalization = ((max_x - min_x) + (max_y - min_y)) / 2
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
    return dists


def loadJSON(path):
    with open(path) as f:
        return json.load(f)


def getLandmarks(landmarksStore, scalingFactor):
    return [[{"x": lm["x"]*scalingFactor, "y": lm["y"]*scalingFactor}
             for lm in landmarkList] for landmarkList in landmarksStore]


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


def isFacingLeft(landmarks):
    # check first frame predictions to see if left wrist is more to the left than left hip
    # left wrist
    wrist = landmarks[0][7]
    # left hip
    hip = landmarks[0][4]
    return wrist['x'] < hip['x']
