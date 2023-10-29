# load state.json and results_kolo_zlute.json
import glob
import pprint

import numpy as np

from utils import *

scalingFactor = 1.6
landmarks_file = 'gt/kolo_cerne.json'
results_files = glob.glob("predictions/*/results_kolo_cerne.json")
landmark_names = ["foot", "heel", "ankle",
                  "knee", "hip", "shoulder", "elbow", "wrist"]
show = False
video_path = 'kolo_cerne.mp4'

keypoint_names = ["left_small_toe", "left_heel", "left_ankle",
                  "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]


def main():

    gt = loadJSON(landmarks_file)
    landmarks = getLandmarks(gt['$landmarksStore'], scalingFactor)
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
