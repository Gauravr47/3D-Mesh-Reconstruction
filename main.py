# main.py
import argparse
import yaml
from pathlib import Path

from scripts.run_colmap import automatic_pipeline
from scripts.convert_vid_to_img import extract_frames, find_video_in_folder

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args(defaults):
    parser = argparse.ArgumentParser(description="3D Mesh Reconstruction Pipeline")

    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dataset_name', type=str, default=defaults['dataset_name'])
    parser.add_argument('--base_data_dir', type=str, default=defaults['base_data_dir'])
    parser.add_argument('--base_results_dir', type=str, default=defaults['base_results_dir'])
    parser.add_argument('--use_gpu', action='store_true' if defaults['use_gpu'] else 'store_false')
    parser.add_argument('--run_nerf', action='store_true' if defaults['run_nerf'] else 'store_false')

    return parser.parse_args()

def main():
    cfg = load_config()
    args = parse_args(cfg)

    dataset = args.dataset_name
    data_dir = Path(args.base_data_dir) / dataset
    result_dir = Path(args.base_results_dir) / dataset

    image_dir = data_dir / "images"
    video_dir = data_dir / "video"
    colmap_workspace = data_dir 
    mesh_output = result_dir / "meshes"
    step_output = mesh_output / "model.step"
    data_is_video = args.data_is_video

    if data_is_video:
        video_file = find_video_in_folder(video_dir)
        if not video_file:
            print(f"[ERROR] No valid input video dataset found")
        extract_frames(video_file, image_dir, 2)
    
    mesh_output.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running pipeline on dataset: {dataset}")
    automatic_pipeline(str(image_dir), str(colmap_workspace))

if __name__ == "__main__":
    main()
