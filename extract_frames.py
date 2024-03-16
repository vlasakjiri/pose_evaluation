import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from utils import *
from sklearn.cluster import KMeans
from scipy.spatial import distance
from glob import glob

np.random.seed(0)

train_videos = glob("D:\\jiriv\\OneDrive\\bikefit videa\\train\\*.mp4")
val_videos = glob("D:\\jiriv\\OneDrive\\bikefit videa\\val\\*.mp4")
NUM_FRAMES = 100


train_zip = zip(train_videos, ["data/train/img"]*len(train_videos))
val_zip = zip(val_videos, ["data/val/img"]*len(val_videos))

for video, out_folder in (list(train_zip) + list(val_zip)):
    basename = os.path.basename(video).split(".")[0]
    # read video file
    cap = cv.VideoCapture(video)

    results_file = f"gt\\{basename}.json"
    results = getLandmarks(loadJSON(results_file)['$landmarksStore'], 1.6)

    # if not facing_left:
    #     keypoint_mapping = flipKeypoints(keypoint_mapping, results)
    #     print("Flipped keypoints", keypoint_mapping)

    predictions = landmarksToArr(results)
    predictions_reshaped = predictions.reshape(len(predictions), -1)
    print(predictions.shape)

    # cluster predictions
    # Create a KMeans instance with n_clusters
    kmeans = KMeans(n_clusters=NUM_FRAMES)

    # Fit the model to your data
    kmeans.fit(predictions_reshaped)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Calculate the distance of each point to its centroid
    distances = np.array([distance.euclidean(point, centroids[label])
                          for point, label in zip(predictions_reshaped, labels)])

    # Initialize list to hold indexes of closest frames
    closest_frames = []

    for i in range(kmeans.n_clusters):
        # Get the index of the closest point in this cluster
        closest_index = np.argmin(distances[labels == i])

        # Get the original index
        original_index = np.where(labels == i)[0][closest_index]

        # Append the original index of the closest frame to the list
        closest_frames.append(original_index)

    print("Original indexes of closest frames to centroids: ", closest_frames)

    frame_idx = 0
    ret, frame = cap.read()
    # get number of frames
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_folder, exist_ok=True)
    assert num_frames == len(predictions)

    # print progress bar
    for _ in tqdm(range(num_frames)):
        if (frame_idx in closest_frames):
            # convert to CIELAB color space
            lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv.inRange(lab, (0, 145, 186), (255, 184, 219), mask)

            # Remove small noise
            kernel = np.ones((3, 3), np.uint8)
            inp_mask = cv.erode(mask, kernel, iterations=1)
            # dilate mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv.dilate(mask, kernel, iterations=2)

            dst = cv.inpaint(frame, mask, 15, cv.INPAINT_NS)
            # prediction = predictions[frame_idx]
            # for point in prediction:
            #     cv.circle(dst, (int(point[0]), int(
            #         point[1])), 5, (255, 0, 0), -1)

            # save inpainted frame set jpg quality to 80
            out_filename = f'{basename}_{frame_idx}.jpg'
            cv.imwrite(os.path.join(out_folder, out_filename), dst, [
                int(cv.IMWRITE_JPEG_QUALITY), 80])
            # cv.imshow("mask", mask)
            # cv.imshow('dst', dst)
            # if cv.waitKey(10) & 0xFF == ord('q'):
            #     break
        ret, frame = cap.read()
        frame_idx += 1
