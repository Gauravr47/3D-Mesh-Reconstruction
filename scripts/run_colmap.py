import os
import shutil
import subprocess
from scripts.error import COLMAPError
from scripts.logger import logger
from scripts.config import get_config
from scripts.colmap_options import ColmapCommand, AutomaticReconstructorOptions, FeatureExtractorOptions, ExhaustiveMatcherOptions, SequentialMatcherOptions, SpatialMatcherOptions, VocabTreeMatcherOptions,update_options
from scripts.image_manager import count_colmap_images_recursive
## Function to get the colmap binary
# Searches colmap if set as env variable
# Other search location
# "~\\vcpkg\\installed\\x64-linux\\tools\\colmap\\colmap")
# "~\\vcpkg\\installed\\x64-windows\\tools\\colmap\\colmap.exe")
# "C:\\vcpkg\\installed\\x64-windows\\tools\\colmap\\colmap.exe")
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

## Entry point to run colmap C++ binary via python subprocess. 
# All inputs are string values
# Format - Command --arg1 val1 --arg2 val2 .... 
def run_colmap(command, args): 
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

## Function to get appropriate options for COLMAP controller command
# AUTOMATIC_RECONSTRUCTOR: AutomaticReconstructorOptions
# FEATURE_EXTRACTOR: FeatureExtractorOptions
# MATCHER: ExhaustiveMatcherOptions
def get_colmap_options_class(command: ColmapCommand):
    return {
        ColmapCommand.AUTOMATIC_RECONSTRUCTOR: AutomaticReconstructorOptions,
        ColmapCommand.FEATURE_EXTRACTOR: FeatureExtractorOptions,
        ColmapCommand.EXHAUSTIVE_MATCHER: ExhaustiveMatcherOptions,
        ColmapCommand.SEQUENTIAL_MATCHER: SequentialMatcherOptions,
        ColmapCommand.SPATIAL_MATCHER: SpatialMatcherOptions,
        ColmapCommand.VOCAB_TREE_MATCHER: VocabTreeMatcherOptions
        # Add more as needed
    }[command]

## Function to override options for COLMAP controller command by user defined CLI options
def get_colmap_override_dict(command: ColmapCommand):
    cfg = get_config()
    return {
        ColmapCommand.AUTOMATIC_RECONSTRUCTOR: cfg.automatic_reconstructor_options,
        ColmapCommand.FEATURE_EXTRACTOR: cfg.feature_extractor_options,
        ColmapCommand.EXHAUSTIVE_MATCHER: cfg.exhaustive_matcher_options,
        ColmapCommand.SEQUENTIAL_MATCHER: cfg.sequential_matcher_options,
        ColmapCommand.SPATIAL_MATCHER: cfg.spatial_matcher_options,
        ColmapCommand.VOCAB_TREE_MATCHER: cfg.vocab_tree_matcher_options
        # Add more as needed
    }[command]

## Implementation Function to run COLMAP controller command
# Based on the command, appropriate options are selected 
# Internally uses run_colmap as an entry point to COLMAP C++ binaries
def run_colmap_impl(command: ColmapCommand): 
    try:
        #Make sure all the options are correct based on command
        #get config
        cfg = get_config()
        options = get_colmap_options_class(command)()
        
        override_options = get_colmap_override_dict(command)
        #Override options if any passed through config yaml
        override_options = override_options |{
            'image_path': cfg.image_dir,
            'workspace_path': cfg.results_dir,
            'database_path': cfg.results_dir+"\\database.db" ,
            'output_path':cfg.results_dir,
            'vocab_tree_path':cfg.vocab_tree_path
        }
        
        if override_options:
            update_options(options, override_options)

        run_colmap(command.value, options.to_cli_args())
    except COLMAPError as e:   
        raise COLMAPError(e)


def generate_sparse():
    cfg = get_config()
    command = ColmapCommand

    command = ColmapCommand.FEATURE_EXTRACTOR
    try:
        run_colmap_impl(command)
    except COLMAPError as e:
        raise COLMAPError(e)
    except Exception as e:
        raise COLMAPError(e)

    if cfg.data_type.lower() == "video":
        command = ColmapCommand.SEQUENTIAL_MATCHER
    elif count_colmap_images_recursive(cfg.image_dir) < 1000: 
        command = ColmapCommand.EXHAUSTIVE_MATCHER
    elif cfg.vocab_tree_path :
        command = ColmapCommand.VOCAB_TREE_MATCHER
    elif cfg.spatial_data_available:
        command = ColmapCommand.SPATIAL_MATCHER
    else:
        command = ColmapCommand.EXHAUSTIVE_MATCHER

    try:
        run_colmap_impl(command)
    except COLMAPError as e:
        raise COLMAPError(e)
    except Exception as e:
        raise COLMAPError(e)
    

