
from glob import glob
import os

from matplotlib.pylab import annotations

from utils import *
import cv2 as cv


image_folder = "frames/"
predictions_folder = "./results/rtmpose-l_8xb512-700e_body8-halpe26-256x192/predictions/"
landmarks_folder = "./gt/"

image_files = glob(image_folder + "*")

keypoint_names = ["left_small_toe", "left_heel", "left_ankle",
                  "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]

annotations = []
images = []
categories = [{
    "id": 1,
    "name": "person"
}]
for id, image_file in enumerate(image_files):
    basename = os.path.basename(image_file).split(".")[0]
    video_name = basename.split("_")[0]
    frame_number = int(basename.split("_")[1])

    # load gt
    gt_file = f"gt\\{video_name}.json"
    gt = getLandmarks(loadJSON(gt_file)['$landmarksStore'], 1.6)
    facing_left = isFacingLeft(gt)
    gt = landmarksToArr(gt)

    # load predictions
    predictions_file = predictions_folder + f"{video_name}.json"
    predictions = loadJSON(predictions_file)
    keypoint_mapping = getKeypointMapping(keypoint_names, predictions)
    print(keypoint_mapping)

    if not facing_left:
        keypoint_mapping = flipKeypoints(keypoint_mapping, predictions)
        print("Flipped keypoints", keypoint_mapping)
    predictions = predictions['instance_info']
    assert len(predictions) == len(gt)
    predictions = predictionsToArr(predictions, keypoint_mapping)

    gt = gt[frame_number]
    predictions = predictions[frame_number]

    # get width and height
    img = cv.imread(image_file)
    height, width, _ = img.shape

    # create json annotations in coco format
    images.append({
        "id": id,
        "file_name": os.path.basename(image_file),
        "height": height,
        "width": width
    })

    # img = cv2.imread(image_file)
    # for point in gt:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    # for point in predictions:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)
    # cv.imshow('frame', img)
    # while True:
    #     if cv.waitKey(10) & 0xFF == 27:
    #         break
