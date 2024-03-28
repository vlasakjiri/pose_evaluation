# load state.json and results_kolo_zlute.json
# %%

import glob
import os
import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *

scalingFactor = 1.6
landmarks_files = glob.glob("gt/*")
landmark_names = ["foot", "heel", "ankle",
                  "knee", "hip", "shoulder", "elbow", "wrist"]
show = False
video_path = "C:\\Users\\jiriv\\OneDrive\\bikefit videa\\val\\10.mp4"

keypoint_names = ["left_small_toe", "left_heel", "left_ankle",
                  "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]


def main():
    data = []
    for model in tqdm(glob.glob("predictions/*")):
        print("Model: " + model)
        model_distances = []
        for image_prediction in tqdm(glob.glob(f"{model}/predictions/*")):
            # print("Image prediction: " + image_prediction)

            # print(predictions)
            # print()
            basename= os.path.basename(image_prediction)
            video_name = basename.split("_")[0]
            frame_number = int(basename.split("_")[-1].split(".")[0])

            landmarks_file = f"gt\\{video_name}.json"
            gt = loadJSON(landmarks_file)
            landmarks = getLandmarks(gt['$landmarksStore'], scalingFactor)
            facing_left = isFacingLeft(landmarks)
            landmarks = landmarksToArr(landmarks)

            results = loadJSON(image_prediction)
            keypoint_mapping = getKeypointMapping(keypoint_names, results)
            # print(keypoint_mapping)
            predictions = results['instance_info']
            # assert len(predictions) == len(landmarks)
            if not facing_left:
                keypoint_mapping = flipKeypoints(keypoint_mapping, results)
                # print("Flipped keypoints", keypoint_mapping)

            predictions = predictionsToArr(predictions, keypoint_mapping)

            # get only the frame we are interested in
            frame_landmarks = landmarks[frame_number]
            frame_landmarks = frame_landmarks[np.newaxis, :]

            distances = calcDistancesArr(frame_landmarks, predictions).squeeze(0)
            model_distances.append(distances)
        
        model_distances = np.mean(model_distances, axis=0)
        distances_dict = dict(zip(landmark_names, model_distances))
        print(distances_dict)
        print()
        data.append({
            "model": model,
            **distances_dict
        })
    return data



if __name__ == "__main__":
    data = main()
    df = pd.DataFrame(data)
    df.to_csv("results.csv", index=False)
    print(df)
    # df = df.drop(columns="video").groupby("model").mean()
    # %%

    # all = df.drop(columns=["foot", "heel"])
    # all["mean"] = all.mean(axis=1)
    # all = all.sort_values(by="mean")
    # all.index = all.index.str.replace("_", "\_")
    # print(all)
    # # %%
    # wholebody = df.copy()
    # wholebody = wholebody[wholebody["foot"] != -1]
    # wholebody["mean"] = wholebody.mean(axis=1)
    # wholebody = wholebody.sort_values(by="mean")
    # wholebody.index = wholebody.index.str.replace("_", "\_")
    # print(wholebody.to_latex(float_format="%.2f"))


# %%
