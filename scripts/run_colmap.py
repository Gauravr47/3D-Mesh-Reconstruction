import os
import shutil
import subprocess
from scripts.error import COLMAPError
from scripts.logger import logger
from scripts.config import get_config
from scripts.colmap_options import ColmapCommand, AutomaticReconstructorOptions, FeatureExtractorOptions, ExhaustiveMatcherOptions

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

def run_colmap(command, args): #General command to colmap controllers via python subprocess. All inputs are strgit status
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

def get_colmap_options_class(command: ColmapCommand):
    return {
        ColmapCommand.AUTOMATIC_RECONSTRUCTOR: AutomaticReconstructorOptions,
        ColmapCommand.FEATURE_EXTRACTOR: FeatureExtractorOptions,
        ColmapCommand.MATCHER: ExhaustiveMatcherOptions,
        # Add more as needed
    }[command]
    
def run_colmap_pipeline(command: ColmapCommand, image_path, output_path, override_options: dict = None): #Command to run any colmap CLI function in COLMAP
    try:
        """Make sure all the options are correct based on command"""
        #get config
        cfg = get_config()
        options = get_colmap_options_class(command)(cfg.image_dir, cfg.data_dir)
        
        #Override options if any passed through config yaml
        if override_options:
            for k, v in override_options.items():
                if hasattr(options, k):
                    setattr(options, k, v)

        run_colmap(command.value, options.to_cli_args())
    except COLMAPError as e:   
        raise COLMAPError(e)


    
