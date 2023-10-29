# load state.json and results_kolo_zlute.json

import glob
import json
import pprint

import cv2
import numpy as np

from utils import *

scalingFactor = 1.6
landmarks_file = 'gt/kolo_cerne.json'
results_file = "myvelofit/kolo_cerne_myvelofit.json"
landmark_names = ["foot", "heel", "ankle",
                  "knee", "hip", "shoulder", "elbow", "wrist"]
show = True
video_path = 'kolo_cerne.mp4'


def main():
    gt = loadJSON(landmarks_file)
    landmarks = getLandmarks(gt['$landmarksStore'], scalingFactor)

    results = loadJSON(results_file)
    predictions = getLandmarks(results['$landmarksStore'], scalingFactor)

    # landmarks and predictions can have different framerates, have to sync them
    landmarks_timestamps = gt['$frameTimestampsStore']
    predictions_timestamps = results['$frameTimestampsStore']

    # cut to shorter one
    last_landmark_timestamp = landmarks_timestamps[-1]
    last_prediction_timestamp = predictions_timestamps[-1]
    end_timestamp = min(last_landmark_timestamp["timestamp"] + last_landmark_timestamp["duration"],
                        last_prediction_timestamp["timestamp"] + last_prediction_timestamp["duration"])

    landmarks_timestamps = [
        timestamp for timestamp in landmarks_timestamps if timestamp["timestamp"] < end_timestamp]
    predictions_timestamps = [
        timestamp for timestamp in predictions_timestamps if timestamp["timestamp"] < end_timestamp]

    landmarks = landmarks[:len(landmarks_timestamps)]
    predictions = predictions[:len(predictions_timestamps)]

    if len(landmarks) < len(predictions):
        landmark_to_prediction_mapping = []
        prediction_idx = 0
        # mapping between gt and predictions keypoints
        for landmark_idx, landmark_timestamp in enumerate(landmarks_timestamps):
            prediction_timestamp = predictions_timestamps[prediction_idx]
            # while they have empty union
            while prediction_timestamp["timestamp"] + prediction_timestamp["duration"] <= landmark_timestamp["timestamp"]:
                prediction_idx += 1
                prediction_timestamp = predictions_timestamps[prediction_idx]
            landmark_to_prediction_mapping.append(prediction_idx)
        predictions = [predictions[prediction_idx]
                       for prediction_idx in landmark_to_prediction_mapping]
    else:
        prediction_to_landmark_mapping = []
        landmark_idx = 0
        # mapping between gt and predictions keypoints
        for prediction_idx, prediction_timestamp in enumerate(predictions_timestamps):
            landmark_timestamp = landmarks_timestamps[landmark_idx]
            # while they have empty union
            while landmark_timestamp["timestamp"] + landmark_timestamp["duration"] <= prediction_timestamp["timestamp"]:
                landmark_idx += 1
                landmark_timestamp = landmarks_timestamps[landmark_idx]
            prediction_to_landmark_mapping.append(landmark_idx)
        landmarks = [landmarks[landmark_idx]
                     for landmark_idx in prediction_to_landmark_mapping]

    assert len(predictions) == len(
        landmarks), f"Number of frames in predictions ({len(predictions)}) and landmarks ({len(landmarks)}) does not match"

    for i, prediction in enumerate(predictions):
        if len(prediction) == 6:
            predictions[i] = [{}, {}] + prediction

    assert len(predictions[0]) == len(
        landmarks[0]), f"Number of landmarks in predictions ({len(predictions[0])}) and landmarks ({len(landmarks[0])}) does not match"

    landmarks = landmarksToArr(landmarks)
    predictions = landmarksToArr(predictions)

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
