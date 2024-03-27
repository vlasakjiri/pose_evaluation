
from glob import glob
import os


from utils import *
import cv2 as cv

split_folder = ".\\data\\train"
image_folder = os.path.join(split_folder, "img\\")
predictions_folder = "./results/rtmpose-l_8xb512-700e_body8-halpe26-256x192/predictions/"
landmarks_folder = "./gt/"
annotations_output = os.path.join(split_folder, "annotations.json")

print("Creating annotations for", image_folder)

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
    print("Processing", basename)

    video_name = basename.split("_")[0]
    frame_number = int(basename.split("_")[-1])

    # load gt
    gt_file = f"gt\\{video_name}.json"
    gt = getLandmarks(loadJSON(gt_file)['$landmarksStore'], 1.6)
    facing_left = isFacingLeft(gt)
    gt = landmarksToArr(gt)

    # load predictions
    predictions_file = predictions_folder + f"{video_name}.json"
    predictions = loadJSON(predictions_file)
    keypoint_mapping = getKeypointMapping(keypoint_names, predictions)

    if not facing_left:
        keypoint_mapping = flipKeypoints(keypoint_mapping, predictions)
        # print("Flipped keypoints", keypoint_mapping)
    predictions = predictions['instance_info']

    if len(predictions) != len(gt):
        print("Different number of frames in predictions and gt")
        print("Predictions:", len(predictions))
        print("GT:", len(gt))
        continue
    # predictions = predictionsToArr(predictions, keypoint_mapping)

    gt = gt[frame_number]
    predictions = predictions[frame_number]["instances"][0]

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
    keypoints = [keypoint + [1] for keypoint in predictions["keypoints"]]
    for idx, gt_keypoint in zip(keypoint_mapping, gt):
        if idx != -1:
            keypoints[idx] = list(gt_keypoint) + [2]

    # create bbox from keypoints min and max

    # xmin = min([keypoint[0] for keypoint in keypoints if keypoint[2] > 0])-padding
    # ymin = min([keypoint[1] for keypoint in keypoints if keypoint[2] > 0])-padding
    # xmax = max([keypoint[0] for keypoint in keypoints if keypoint[2] > 0])+padding
    # ymax = max([keypoint[1] for keypoint in keypoints if keypoint[2] > 0])+padding

    # bbox = [xmin, ymin, xmax, ymax]
    bbox = predictions["bbox"][0]
    # # padding = (bbox[2] - bbox[0]) * 0.05
    padding = 0

    # print(bbox)
    bbox[0] = min(bbox[0], min([point[0] for point in gt]) - padding)
    bbox[1] = min(bbox[1], min([point[1] for point in gt]) - padding)
    bbox[2] = max(bbox[2], max([point[0] for point in gt]) + padding)
    bbox[3] = max(bbox[3], max([point[1] for point in gt]) + padding)

    # convert from [x,y,x2, y2] to [x,y,w,h]
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]

    area = bbox[2] * bbox[3]

    annotations.append({
        "id": id,
        "image_id": id,
        "category_id": 1,
        "keypoints": keypoints,
        "num_keypoints": len(keypoints),
        "bbox": bbox,
        "area": area,
        "iscrowd": 0
    })
    # print(annotations)

    # img = cv2.imread(image_file)
    # for point in gt:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    # for point in predictions:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)
    # cv.imshow('frame', img)
    # while True:
    #     if cv.waitKey(10) & 0xFF == 27:
    #         break

dataset = {"images": images, "annotations": annotations,
           "categories": categories}

with open(annotations_output, 'w') as f:
    json.dump(dataset, f)
