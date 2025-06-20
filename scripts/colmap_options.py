from dataclasses import dataclass, fields, asdict, field, is_dataclass
from typing import List, Union, Optional, Type
from enum import Enum
from scripts.error import COLMAPError

class ColmapCommand(Enum):
    AUTOMATIC_RECONSTRUCTOR = "automatic_reconstructor"
    FEATURE_EXTRACTOR = "feature_extractor"
    MATCHER = "exhaustive_matcher"
    MAPPER = "mapper"
    IMAGE_UNDISTRORTER = "image_undistorter"
    PATCH_MATCH = "patch_match_stereo"
    FUSION = "stereo_fusion"
    MESHER = "poisson_mesher"

class DataType(str, Enum):
    INDIVIDUAL = "individual"
    VIDEO = "video"
    INTERNET = "internet"

class Quality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class CameraModel(str, Enum):
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"

class MesherType(str, Enum):
    POISSON = "poisson"
    DELAUNAY = "delaunay"

## L1-normalizes each descriptor followed by element-wise square rooting.
## This normalization is usually better than standard L2-normalization.
## See "Three things everyone should know to improve object retrieval",
## Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
class DescriptorNormalization(str, Enum):
    L1_ROOT = "l1_root"
    L2 = "l2"

@dataclass
class ColmapOptions: ## base colmap option class to be inherited by other option data classes

    
    def to_cli_args(self) -> List[str]:
        args = []

        def process_dataclass(obj, prefix=""):
            try:
                for f in fields(obj):
                    val = getattr(obj, f.name)
                    if val is None:
                        continue

                    key = f"--{prefix}{f.name}" if prefix else f"--{f.name}"

                    if is_dataclass(val):
                        # Recursively process nested dataclass
                        process_dataclass(val, prefix=f"{f.name}.")
                    elif isinstance(val, bool):
                        args.extend([key, "true" if val else "false"])
                    elif isinstance(val, list):
                        args.extend([key, ",".join(str(v) for v in val)])
                    else:
                        args.extend([key, str(val)])
            except Exception as e:
                raise COLMAPError(e)

        process_dataclass(self)
        return args


     # Validate enum inputs if they were given as strings
    def ensure_enum(self, value, enum_type: Type):
        if isinstance(value, enum_type):
            return value
        elif isinstance(value, str):
            val_lower = value.lower()
            for member in enum_type:
                if isinstance(member.value, str) and member.value.lower() == val_lower:
                    return member.value
            raise ValueError(f"Invalid value '{value}' for {enum_type.__name__}")
        else:
            raise TypeError(f"Expected {enum_type.__name__}, got {type(value)}")

   
    def post_init_enum_validation(self, field_enum_map):
        for field_name, enum_type in field_enum_map.items():
            val = getattr(self, field_name)
            setattr(self, field_name, self.ensure_enum(val, enum_type))


## Automatic reconstror colmap option class 
# Inherits Colmap option classes
@dataclass
class AutomaticReconstructorOptions(ColmapOptions): 
    project_path: Optional[str] = None
    workspace_path: Optional[str] = None  ## Root path to folder which contains the results. results_dir/dataset_name
    image_path: Optional[str] = None ## Root path to folder which contains the images. data_dir/dataset_name/images
    
    ## Optional root path to folder which contains image masks. For a given image,
    ## the corresponding mask must have the same sub-path below this root as the
    ## image has below image_path. The filename must be equal, aside from the
    ## added extension .png. For example, for an image image_path/abc/012.jpg, the
    ## mask would be mask_path/abc/012.jpg.png. No features will be extracted in
    ## regions where the mask image is black (pixel intensity value 0 in
    ## grayscale).
    mask_path: Optional[str] = None
    vocab_tree_path: Optional[str] = None

    # Settings
    random_seed: int = 0
    log_to_stderr: bool = True
    log_level: int = 0
    data_type: str = "individual"  # choices: individual, video, internet
    quality: str = "high"          # choices: low, medium, high, extreme
    camera_model: Optional[str] = "SIMPLE_RADIAL"
    single_camera: bool = False
    single_camera_per_folder: bool = False
    camera_params: Optional[str] = None

    # Pipeline stage toggles
    extraction: bool = True
    matching: bool = True
    sparse: bool = True
    dense: bool = True

    # Meshing options
    mesher: Optional[str] = "poisson"        # choices: poisson, delaunay

    # Performance
    num_threads: int = -1
    use_gpu: bool = True
    gpu_index: List[int] = field(default_factory=lambda: [-1])

    def __post_init__(self):
        self.post_init_enum_validation({
            "data_type": DataType,
            "quality" : Quality,
            "camera_model": CameraModel,
            "mesher": MesherType
        })

## Image Reader colmap option class 
# Inherits Colmap option classes
@dataclass 
class ImageReaderOptions(ColmapOptions):
    database_path: Optional[str] = None ## Path to database in which to store the extracted data.
    image_path: Optional[str] = None  ## Root path to folder which contains the images.
    
    ## Optional root path to folder which contains image masks. For a given image,
    ## the corresponding mask must have the same sub-path below this root as the
    ## image has below image_path. The filename must be equal, aside from the
    ## added extension .png. For example, for an image image_path/abc/012.jpg, the
    ## mask would be mask_path/abc/012.jpg.png. No features will be extracted in
    ## regions where the mask image is black (pixel intensity value 0 in
    ## grayscale).
    mask_path: Optional[str] = None

    ## Optional list of images to read. The list must contain the relative path
    ## of the images with respect to the image_path.
    image_list: Optional[List[str]] = None
    camera_model: str = "SIMPLE_RADIAL" ## Name of the camera model.
    single_camera: bool = False   ## Whether to use the same camera for all images.
    single_camera_per_folder: bool = False  ## Whether to use the same camera for all images in the same sub-folder.
    single_camera_per_image: bool = False ## Whether to use a different camera for each image.
        
    ## Whether to explicitly use an existing camera for all images. Note that in
    ## this case the specified camera model and parameters are ignored.
    existing_camera_id: int = -1  # Assuming kInvalidCameraId = -1
    
    ## Manual specification of camera parameters. If empty, camera parameters
    ## will be extracted from EXIF, i.e. principal point and focal length.
    camera_params: Optional[str] = None

     
    ## If camera parameters are not specified manually and the image does not
    ## have focal length EXIF information, the focal length is set to the
    ## value `default_focal_length_factor * max(width, height)`.
    default_focal_length_factor: float = 1.2

    ## Optional path to an image file specifying a mask for all images. No
    ## features will be extracted in regions where the mask is black (pixel
    ## intensity value 0 in grayscale).
    camera_mask_path: Optional[str] = None

    def __post_init__(self):
        self.post_init_enum_validation({
            "camera_model": CameraModel,
        })

@dataclass
class SiftExtractionOptions(ColmapOptions):
    num_threads: int = -1  ##  Number of threads for feature extraction.
    use_gpu: bool = True  ## Whether to use the GPU for feature extraction.

    max_image_size: int = 3200  ## Maximum image size, otherwise image will be down-scaled.
    max_num_features: int = 8192  ## Maximum number of features to detect, keeping larger-scale features.

    first_octave: int = -1  ## First octave in the pyramid, i.e. -1 upsamples the image by one level.
    num_octaves: int = 4  ## Number of octaves.
    octave_resolution: int = 3  ## Number of levels per octave

    peak_threshold: float = 0.0066666666666666671  ## Peak threshold for detection.
    edge_threshold: float = 10.0  ## Edge threshold for detection.

    estimate_affine_shape: bool = False  ## Estimate affine shape of SIFT features in the form of oriented ellipses as opposed to original SIFT which estimates oriented disks.
    max_num_orientations: int = 2  ## Maximum number of orientations per keypoint if not estimate_affine_shape.
    upright: bool = False  ## Fix the orientation to 0 for upright features.

    ## Domain-size pooling parameters. Domain-size pooling computes an average
    ## SIFT descriptor across multiple scales around the detected scale. This was
    ## proposed in "Domain-Size Pooling in Local Descriptors and Network
    ## Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
    ## outperform other SIFT variants and learned descriptors in "Comparative
    ## Evaluation of Hand-Crafted and Learned Local Features", Schönberger,
    ## Hardmeier, Sattler, Pollefeys, CVPR 2016.
    domain_size_pooling: bool = False  
    dsp_min_scale: float = 0.16666666666666666  ## --SiftExtraction.dsp_min_scale
    dsp_max_scale: float = 3.0  ## --SiftExtraction.dsp_max_scale
    dsp_num_scales: int = 10  ## --SiftExtraction.dsp_num_scales
    gpu_index: List[int] = field(default_factory=lambda: [-1])  ## Index of the GPU used for feature extraction. For multi-GPU extraction, you should separate multiple GPU indices by comma, e.g., "0,1,2,3".

## Feature Extractor colmap option class 
# Inherits Colmap option classes
@dataclass
class FeatureExtractorOptions(ColmapOptions):
    descriptor_normalization: str ="l1_root"
    ImageReader: ImageReaderOptions = ImageReaderOptions()
    SiftExtraction: SiftExtractionOptions = SiftExtractionOptions()
    log_to_stderr: bool = True
    random_seed: int = 0  ## random_seed
    log_to_stderr: bool = True  ## log to stderr
    log_level: int = 0  ## log level
    project_path: Optional[str] = None  ## path to root of the project i.e results/dataset_name
    database_path: Optional[str] = None  ## path to root of the project i.e results/dataset_name
    image_path: Optional[str] = None  ## path to image folder i.e data/dataset_name/images
    camera_mode: int = -1  ## --camera_mode
    image_list_path: Optional[str] = None 

    def __post_init__(self):
        self.post_init_enum_validation({
            "descriptor_normalization": DescriptorNormalization
        })


@dataclass
class ExhaustiveMatcherOptions(ColmapOptions):
    log_to_stderr: bool = True
