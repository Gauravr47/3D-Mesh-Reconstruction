from dataclasses import dataclass, fields, asdict, field
from typing import List, Union, Optional, Type
from enum import Enum

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

@dataclass
class ColmapOptions: #base colmap option class to be inherited by other option data classes
    image_path: str = ""
    workspace_path: str = ""

    def to_cli_args(self) -> List[str]: #return a list of args by iterating the class
        args = []
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue

            key = f"--{f.name}"
            # Special cases
            if isinstance(val, bool):
                args.extend([key, "true" if val else "false"])
            elif isinstance(val, list):
                args.extend([key, ",".join(str(v) for v in val)])
            else:
                args.extend([key, str(val)])
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

    
#Inherit Colmap option classes
@dataclass
class AutomaticReconstructorOptions(ColmapOptions):
    project_path: Optional[str] = None
    workspace_path: Optional[str] = None
    image_path: Optional[str] = None
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

@dataclass
class FeatureExtractorOptions(ColmapOptions):
    log_to_stderr: bool = True


@dataclass
class ExhaustiveMatcherOptions(ColmapOptions):
    log_to_stderr: bool = True
