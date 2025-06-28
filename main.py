# main.py
import argparse
import os
import torch

from pathlib import Path

from scripts.colmap_options import ColmapCommand
from scripts.run_colmap import run_colmap_impl, generate_sparse, generate_dense
from scripts.image_manager import extract_frames, find_video_in_folder
from scripts.error import PipelineError, COLMAPError, Open3DError
from scripts.logger import logger
from scripts.config import load_config, get_config


def parse_args(defaults): #function to parse CLI parameters
    parser = argparse.ArgumentParser(description="3D Mesh Reconstruction Pipeline")

    parser.add_argument('--config_path', type=str, default='configs')
    parser.add_argument('--config_name', type=str, default='config.yaml')

    #override variables
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--base_data_dir', type=str, default=None)
    parser.add_argument('--base_results_dir', type=str, default=None)
    parser.add_argument('--run_nerf', action='store_true', help="Run NeRF for creating a more accurate mesh")
    parser.add_argument('--data_type', type=str,help="# choices: individual, video, internet", default=None)

    #group
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_gpu', dest='use_gpu', action='store_true', help="Force GPU usage if available")
    group.add_argument('--no_gpu', dest='use_gpu', action='store_false', help="Force CPU even if GPU is available")
    parser.set_defaults(use_gpu=None)
    return parser.parse_args()

def main():
    logger.info("Starting pipeline")
    
    args = parse_args(None)

    #load config and realted parameters
    load_config(args.config_path, args.config_name)
    cfg = get_config()
    
    if cfg.use_gpu:
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            for i in range(num_cuda_devices):
                device_name = torch.cuda.get_device_name(i)
                device_properties = torch.cuda.get_device_properties(i)
                logger.info(f"Device Index: {i}")
                logger.info(f"Device Name: {device_name}")
                logger.info(f"Total Memory: {device_properties.total_memory / (1024**3):.2f} GB")
                logger.info(f"Compute Capability: {device_properties.major}.{device_properties.minor}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        logger.info(" CUDA enabled for feature extraction / matching")

    try:
        if cfg.data_type == "video":
            video_file = find_video_in_folder(cfg.video_dir)
            if not video_file:
                raise PipelineError(" No valid input video dataset found")
            extract_frames(video_file, cfg.image_dir, 2)
    except PipelineError as e:
        logger.error(" No valid input video dataset found")
    
    Path(cfg.mesh_dir).mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f" Running pipeline on dataset: {cfg.dataset_name}")
        generate_dense()
    except COLMAPError as e:
        logger.error(f"COLMAP failed : {e}")

if __name__ == "__main__":
    main()
