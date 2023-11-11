# load state.json and results_kolo_zlute.json
# %%

import glob
import os
import pprint

import numpy as np
import pandas as pd

from utils import *

scalingFactor = 1.6
landmarks_files = glob.glob("gt/*.json")
landmark_names = ["foot", "heel", "ankle",
                  "knee", "hip", "shoulder", "elbow", "wrist"]
show = False
video_path = '1-aero.mp4'

keypoint_names = ["left_small_toe", "left_heel", "left_ankle",
                  "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]


def main():
    data = []
    for landmarks_file in landmarks_files:
        print("Landmarks file: " + landmarks_file)

        gt = loadJSON(landmarks_file)
        landmarks = getLandmarks(gt['$landmarksStore'], scalingFactor)
        facing_left = isFacingLeft(landmarks)
        print("Facing left: " + str(facing_left))

        landmarks = landmarksToArr(landmarks)

        base_filename = os.path.basename(landmarks_file).split(".")[0]

        results_files = glob.glob(
            f"results/*/predictions/{base_filename}.json")

        for results_file in results_files:
            model_name = results_file.split('\\')[1]
            # print("Model: " + model_name)
            # print("Results file: " + results_file)

            results = loadJSON(results_file)
            keypoint_mapping = getKeypointMapping(keypoint_names, results)

            predictions = results['instance_info']
            assert len(predictions) == len(landmarks)
            if not facing_left:
                keypoint_mapping = flipKeypoints(keypoint_mapping, results)

            predictions = predictionsToArr(predictions, keypoint_mapping)

            distances = calcDistancesArr(landmarks, predictions)

            landmarks_distances = np.mean(distances, axis=0)

            distances_dict = dict(zip(landmark_names, landmarks_distances))

            data.append({
                "model": model_name,
                "video": video_path,
                **distances_dict
            })

            # print("Landmarks distances: \n" +
            #       pprint.pformat(distances_dict, indent=4))
            # print("Total avg distance: " +
            #       str(np.mean(landmarks_distances, where=landmarks_distances != -1)))

            # print()
            if show:
                visualizeLandmarks(landmarks, predictions, video_path)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = main()
    df= df.drop(columns="video").groupby("model").mean()
    # %%

    all = df.drop(columns=["foot","heel"])
    all["mean"] = all.mean(axis=1)
    all.sort_values(by="mean")
    # %%
    wholebody = df.copy()
    wholebody = wholebody[wholebody["foot"] != -1]
    wholebody["mean"] = wholebody.mean(axis=1)
    wholebody.sort_values(by="mean")


# %%
