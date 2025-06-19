import os
import shutil
import subprocess
from scripts.error import COLMAPError
from scripts.logger import logger

def find_colmap():
    colmap = shutil.which("colmap")
    if colmap:
        return colmap

    env_path = os.environ.get("COLMAP_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    guesses = [
        os.path.expanduser("~\\vcpkg\\installed\\x64-linux\\tools\\colmap\\colmap"),
        os.path.expanduser("~\\vcpkg\\installed\\x64-windows\\tools\\colmap\\colmap.exe"),
        os.path.expanduser("C:\\vcpkg\\installed\\x64-windows\\tools\\colmap\\colmap.exe"),
    ]
    for path in guesses:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError("COLMAP binary not found. Set COLMAP_PATH or add to system PATH.")

def run_colmap(command, args):
    """Run COLMAP with the given command and arguments."""
    try:
        colmap_bin = find_colmap()
    except FileNotFoundError as e:
        raise COLMAPError(e)
    
    cmd = [colmap_bin, command] + args
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP command failed with exit code {e.returncode}")
        raise COLMAPError(str(e)) from e

def automatic_pipeline(image_path, output_path):
    try:
        """Run full SfM+MVS reconstruction."""
        run_colmap("automatic_reconstructor", [
            "--image_path", image_path,
            "--workspace_path", output_path,
            "--dense", "1", "--use_gpu","true", 
            "--num_threads", "-1",
            "gpu_index", "0,1"
        ])
    except COLMAPError as e:
        
        raise COLMAPError(e)
    
def exhaustive_matcher(image_path, output_path):
    try:
        """Exhaustive matching."""
        run_colmap("exhaustive_matcher", [
            "--image_path", image_path,
            "--workspace_path", output_path,
            "--dense", "1", "--use_gpu","1", "--num_threads","-1"
        ])
    except COLMAPError as e:
        
        raise COLMAPError(e)