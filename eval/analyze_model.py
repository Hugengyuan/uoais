import argparse
import glob
import multiprocessing as mp
import os
import cv2
import imageio
import time
import random
import numpy as np
# constants
from detectron2.engine import DefaultPredictor
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)

from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor

from utils import *
from foreground_segmentation.model import Context_Guided_Network

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter our background instances or not"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to forground segmentation weight"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./sample_data",
        help="path to the OSD dataset"
    )
    return parser


if __name__ == "__main__":

    # UOAIS-Net
    args = get_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    logger.info("Parameter Count:\n" + parameter_count_table(instantiate(cfg.model), max_depth=5))
  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # Load OSD dataset
    rgb_img = cv2.imread("{}/image_color/1.png".format(args.dataset_path))
    rgb_img = cv2.resize("{}/disparity/1.png".format(args.dataset_path))
    depth_img = imageio.imread(depth_path)
    depth_img = normalize_depth(depth_img)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)

    start_time = time.time()
  
    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

    end_time = time.time()

    print(f"predict time: {end_time - start_time}")
        


def do_parameter(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    

def do_time(cfg)
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))
