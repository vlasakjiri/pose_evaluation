import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from utils import *
from sklearn.cluster import KMeans
from scipy.spatial import distance


keypoint_names = ["left_small_toe", "left_heel", "left_ankle",
                  "left_knee", "left_hip", "left_shoulder", "left_elbow", "left_wrist"]

filename = 'kolo_cerne.mp4'
out_folder = "frames"
# read video file
cap = cv.VideoCapture(filename)

results_file = "results\\rtmpose-l_8xb512-700e_body8-halpe26-256x192\\predictions\\kolo_cerne.json"
results = loadJSON(results_file)
keypoint_mapping = getKeypointMapping(keypoint_names, results)

print(keypoint_mapping)
predictions = results['instance_info']
# if not facing_left:
#     keypoint_mapping = flipKeypoints(keypoint_mapping, results)
#     print("Flipped keypoints", keypoint_mapping)

predictions = predictionsToArr(predictions, keypoint_mapping)
predictions_reshaped = predictions.reshape(len(predictions), -1)
print(predictions.shape)

# cluster predictions
# Create a KMeans instance with n_clusters
kmeans = KMeans(n_clusters=40)

# Fit the model to your data
kmeans.fit(predictions_reshaped)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Calculate the distance of each point to its centroid
distances = np.array([distance.euclidean(point, centroids[label]) for point, label in zip(predictions_reshaped, labels)])

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

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# print progress bar
for _ in tqdm(range(num_frames)):
    if(frame_idx in closest_frames):
        # convert to CIELAB color space
        lab = cv.cvtColor(frame, cv.COLOR_RGB2LAB)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.inRange(lab, (0, 31, 137), (255, 100, 216), mask)

        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        inp_mask = cv.erode(mask, kernel, iterations=1)
        # dilate mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=1)

        dst = cv.inpaint(frame, mask, 15, cv.INPAINT_NS)

        # save inpainted frame set jpg quality to 80
        cv.imwrite(os.path.join(out_folder, f' {filename}_{frame_idx}.jpg'), dst, [
                int(cv.IMWRITE_JPEG_QUALITY), 80])
        # cv.imshow("mask", mask)
        # cv.imshow('dst', dst)
        # if cv.waitKey(10) & 0xFF == ord('q'):
        #     break
    ret, frame = cap.read()
    frame_idx += 1
