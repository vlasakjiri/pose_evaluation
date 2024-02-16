
from glob import glob

image_folder = "frames/"
predictions_folder = "results/rtmpose-l_8xb512-700e_body8-halpe26-256x192/predictions/"
landmarks_folder = "gt/"

image_files = glob(image_folder + "*")

for image_file in image_files:
    