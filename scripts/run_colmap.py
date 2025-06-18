import os
import shutil
import subprocess

def find_colmap():
    colmap = shutil.which("colmap")
    if colmap:
        return colmap

    env_path = os.environ.get("COLMAP_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    guesses = [
        os.path.expanduser("~/vcpkg/installed/x64-linux/tools/colmap/colmap"),
        os.path.expanduser("~/vcpkg/installed/x64-windows/tools/colmap/colmap.exe"),
    ]
    for path in guesses:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError("COLMAP binary not found. Set COLMAP_PATH or add to system PATH.")

def run_colmap(command, args):
    """Run COLMAP with the given command and arguments."""
    colmap_bin = find_colmap()
    cmd = [colmap_bin, command] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def automatic_pipeline(image_path, output_path):
    """Run full SfM+MVS reconstruction."""
    run_colmap("automatic_reconstructor", [
        "--image_path", image_path,
        "--workspace_path", output_path,
        "--workspace_format", "COLMAP",
        "--dense", "1"
    ])
