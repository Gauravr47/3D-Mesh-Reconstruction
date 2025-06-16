# 3D-Mesh-Reconstruction
A pipeline that reconstructs watertight 3D meshes from a set of calibrated RGB images using modern CV and geometry techniques.
# 🏗️ 3D Reconstruction Pipeline: COLMAP + NeRF + Open3D (Windows)

This repository documents a full 3D reconstruction and mesh generation pipeline using:

- **COLMAP** for Structure-from-Motion (SfM) and Multi-View Stereo (MVS)
- **NeRF** for neural rendering and depth recovery
- **Open3D** for point cloud processing and mesh generation
- **Python + C++ hybrid architecture**, accelerated with CUDA on Windows

The pipeline supports both **unordered image collections** and **video inputs**, with the goal of enabling side-by-side evaluation of traditional MVS reconstruction and learning-based methods like NeRF and Gaussian Splatting.

---

## 📌 System Configuration

| Component  | Details                      |
| ---------- | ---------------------------- |
| OS         | Windows 10/11 (x64)          |
| GPU        | NVIDIA RTX (3070 or higher)  |
| CUDA       | 12.6 (native), 11.8 (runtime)|
| Python     | 3.10 via `conda`             |
| COLMAP     | Built from source with `vcpkg` |
| NeRF Tools | PyTorch-based, installed via `pip` and `conda` |

---

## 🔄 Pipeline Overview

![Mesh Reconstruction Flow Diagram](./docs/Mesh%20Reconstruction%20flow.png)

> This diagram summarizes the full pipeline — from data input through to visualization and validation.

---

## 🔧 Key Steps

### 1. 📂 **Data Formatting**
- Accepts either image folders or raw videos.
- Automatically extracts frames, creates COLMAP-compatible folders and databases.

### 2. 🧠 **MVS & SfM (COLMAP)**
- Configurable feature matching (sequential, exhaustive, vocabulary, spatial).
- GPU-accelerated dense reconstruction via COLMAP + CUDA.

### 3. 📐 **Point Cloud Extraction**
- Dense depth + normals fused into point cloud.
- Optionally export sparse model and poses for debugging or visualization.

### 4. 🧵 **NeRF Mesh Refinement**
- Trains a NeRF model on registered views.
- Renders depth or novel-view images for additional geometry.

### 5. 🧊 **Mesh Generation (Open3D)**
- Combines COLMAP and NeRF point clouds.
- Applies ICP alignment, normal estimation, outlier removal.
- Generates mesh using Poisson or Ball Pivoting.

### 6. ✅ **Validation & Metrics**
- Geometry metrics: holes, manifold issues, signed distance.
- Image metrics: PSNR, SSIM, LPIPS between NeRF renders and ground truth.
- Logs computation and alignment performance.

### 7. 🖼️ **Visualization**
- Renders and compares NeRF and COLMAP-based meshes.
- Supports export of images, 3D viewers, and combined visualizations.

---

## 📁 Folder Structure (Suggested)
project-root/
 ├── cpp/ #C++ files  
 ├── data/ # Input videos, images, COLMAP workspace  
 ├── logs/ # Metrics and debug outputs  
 ├── nerf/ # Trained NeRF checkpoints  
 ├── scripts/ # Python scripts for each stage  
 ├── results/ # Final outputs  
 ├── notebooks/ #jupytor notebooks   
 ├── vis/ #visualization  
 └── README.md # You are here  

## 📈 Example Use Cases

- Benchmark NeRF vs COLMAP on indoor/outdoor scenes.
- Improve COLMAP meshes with NeRF-generated depth.
- Build photorealistic meshes from smartphone videos.
- Explore point cloud alignment and hybrid fusion techniques.

---

## 📬 Want to Contribute?

Feel free to file issues or suggestions for:
- Additional NeRF variants (e.g., Instant-NGP, Gaussian Splatting)
- Better alignment/fusion techniques
- Integration with Blender, Unity, etc.

---

Thanks for checking out the project!  
— _Gaurav_
