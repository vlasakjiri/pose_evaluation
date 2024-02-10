import glob
import json_tricks as json
from typing import Dict
import mmengine
import os

from numpy import mean
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases


det_config = "configs/rtmdet_nano_320-8xb32_coco-person.py"
det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"

models = [
    ### RTMPose ###
    # body8
    # ("rtmpose-t_8xb256-420e_body8-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth"),

    # ("rtmpose-s_8xb256-420e_body8-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth"),

    # ("rtmpose-m_8xb256-420e_body8-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"),

    # ("rtmpose-l_8xb256-420e_body8-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),

    # ("rtmpose-m_8xb256-420e_body8-384x288",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth"),

    # ("rtmpose-l_8xb256-420e_body8-384x288",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth"),

    # # Halpe
    # ("rtmpose-t_8xb1024-700e_body8-halpe26-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth"),

    # ("rtmpose-s_8xb1024-700e_body8-halpe26-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth"),

    ("rtmpose-m_8xb512-700e_body8-halpe26-256x192",
     "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"),

    # ("rtmpose-l_8xb512-700e_body8-halpe26-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth"),

    # ("rtmpose-m_8xb512-700e_body8-halpe26-384x288",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth"),

    # # Wholebody
    # ("rtmpose-m_8xb64-270e_coco-wholebody-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"),

    # ("rtmpose-l_8xb64-270e_coco-wholebody-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth"),

    # # SIMCC VipNAS
    # ("simcc_vipnas-mbv3_8xb64-210e_coco-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth"),

    # # YOLOPOSE
    # ("yoloxpose_m_8xb32-300e_coco-640",
    #  "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth"),

    # # HRNet
    # ("td-hm_hrnet-w32_8xb64-210e_coco-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"),

    # RTMW
    # ("rtmw-x_8xb704-270e_cocktail13-256x192",
    #  "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.pth")





]

output_root = "results"

inputs = glob.glob("../bikefit_videa/*")

device = "cuda:0"
save_visulization = True
save_predictions = True

draw_heatmap = False
skeleton_style = 'mmpose'

draw_bbox = True
show_kpt_idx = False
show = False
kpt_thr = 0.3
det_cat_id = 0
bbox_thr = 0.3
nms_thr = 0.3


def main():
    for pose_config, pose_checkpoint in models:
        inferencer = MMPoseInferencer(pose2d=pose_config, pose2d_weights=pose_checkpoint, det_model=det_config,
                                      det_weights=det_checkpoint, device=device, det_cat_ids=0)

        for input in inputs:
            base_input = os.path.basename(input).split(".")[0]
            out_dir = os.path.join(
                output_root, os.path.splitext(os.path.basename(pose_config))[0])
            results = inferencer(inputs=input, out_dir=out_dir,
                                 draw_bbox=True, vis_out_dir=None)

            prediction_file = os.path.join(
                out_dir, "predictions", base_input + ".json")
            if os.path.exists(prediction_file):
                print("Skipping prediction file", prediction_file)
                continue
            # log inference times
            timer = mmengine.Timer()
            times = []
            meta_info = inferencer.inferencer.model.dataset_meta
            instance_info = []
            for frame_id, result in enumerate(results):
                # TODO: custom prediction saving
                predictions = result["predictions"][0]
                instance_info.append({
                    "frame_id": frame_id+1,
                    "instances": predictions
                })

                times.append(timer.since_last_check())
            obj = {"meta_info": meta_info,
                   "instance_info": instance_info}
            with open(prediction_file, "w") as f:
                json.dump(obj, f, indent='\t')

            # create file with inference times
            with open(os.path.join(out_dir, base_input + "_times.txt"), "w") as f:
                f.write(str(mean(times)))


main()
