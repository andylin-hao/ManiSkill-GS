from dataclasses import dataclass, field
from typing import List, Optional
import torch
import os
from pathlib import Path
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
import numpy as np
import sapien.core as sapien
from mani_skill_gs.utils.utils_gs import matrix_to_quaternion

# base config
@dataclass
class BaseConfig:
    # name
    name: str
    # pose on the maniskill scene
    p: List[float] = field(default_factory=lambda: [0,0,0])
    q: List[float] = field(default_factory=lambda: [1,0,0,0])
    # offset on the gs space
    offset: List[float] = field(default_factory=lambda: [0,0,0])
    # gs path
    gs_path: Optional[str] = None
    # down sample
    downsample: int = 1
    
    def __post_init__(self):
        # convert list to tensor
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                self.__dict__[key] = torch.tensor(value)

# robot configs
@dataclass
class RobotConfig(BaseConfig):
    """Robot Configuration"""
    # random qpos 
    random_qpos: float = 0.00
    # files
    urdf_path: Optional[str] = None

    def __post_init__(self):
        # asset files
        if self.gs_path is None:
            self.gs_path = str(ROBOT_ASSETS_DIR / self.name / "parts")
            assert os.path.isdir(self.gs_path), f"Robot GS path {self.gs_path} should be a directory."
        if self.urdf_path is None:
            self.urdf_path = str(ROBOT_ASSETS_DIR / self.name / "fr3_wristcam.urdf")
            assert os.path.exists(self.urdf_path), f"Robot URDF path {self.urdf_path} not found."
        # super init
        super().__post_init__()


@dataclass
class ObjectConfig(BaseConfig):
    """Object Configuration"""
    # random perturbation for the initial pose
    p_random: List[float] = field(default_factory=lambda: [0,0,0])
    # scale
    mesh_scale: List[float] = field(default_factory=lambda: [1,1,1])
    gs_scale: List[float] = field(default_factory=lambda: [1,1,1])
    # files
    visual_file: Optional[str] = None
    collision_file: Optional[str] = None
    # physics properties
    density: int = 1000
    mass: float = 0.3
    com_x: float = 0.0
    com_y: float = 0.0
    com_z: float = 0.0
    dynamic_friction: float = 1.0
    static_friction: float = 1.0
    restitution: float = 0.0
    patch_radius: float = 0.1
    min_patch_radius: float = 0.1
    contact_offset: float = 0.01

    def __post_init__(self):
        # asset files
        obj_dir = self.name.split("/")[-1]
        if self.visual_file is None:
            self.visual_file = str(OBJECT_ASSETS_DIR / obj_dir / "mesh_w_vertex_color_abs.obj")
        if self.collision_file is None:
            self.collision_file = str(OBJECT_ASSETS_DIR / obj_dir / "mesh_w_vertex_color_abs.obj")
        if self.gs_path is None:
            self.gs_path = str(OBJECT_ASSETS_DIR / obj_dir / "object_3dgs_abs.ply")
        # super init
        super().__post_init__()

@dataclass
class TableConfig(BaseConfig):
    """Table Configuration"""

    def __post_init__(self):
        # asset files
        if self.gs_path is None:
            self.gs_path = str(OBJECT_ASSETS_DIR / self.name / "table_abs.ply")
        # super init
        super().__post_init__()


# assets paths
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
ASSETS_DIR = _project_root / "assets"
OBJECT_ASSETS_DIR = ASSETS_DIR / "object"
ROBOT_ASSETS_DIR = ASSETS_DIR / "robot"

# TODO: merge into the robot config
# default qpos for franka research3 
FR3_DEFAULT_CFG = {
    "fr3_joint1": 0,
    "fr3_joint2": 2.59358403e-01,
    "fr3_joint3": 0,
    "fr3_joint4": -2.28914773e+00,
    "fr3_joint5": 0,
    "fr3_joint6": 2.51473323e+00,
    "fr3_joint7": 7.85539902e-01,
    "fr3_finger_joint1": 0.04,
    "fr3_finger_joint2": 0.04,
}

# robot configs
ROBOT_CONFIGS = {
    "my_franka": RobotConfig(
        name="my_franka",
        p=[0, 0, 0],
        q=[1, 0, 0, 0],
        offset=[0.0, 0.0, 0.02],
        downsample=1,
    ),
}

# object configs
OBJECT_CONFIGS = {
    "cube": ObjectConfig(
        name="object/cube",
        p=[0.4, 0.2, 0.02],
        q=[1, 0, 0, 0],
        p_random=[0.1, 0.05, 0],
        offset=[0, 0, 0],
        mesh_scale=[1, 1, 1],
        gs_scale=[1, 1, 1],
        downsample=1,
    ),
    "plate": ObjectConfig(
        name="object/plate",
        p=[0.45, 0, 0.02],
        q=[1, 0, 0, 0],
        p_random=[0.05, 0, 0],
        offset=[-0.005, -0.005, 0.00],
        mesh_scale=[1, 1, 1],
        gs_scale=[0.96, 1.00, 0.96],
        downsample=1,
    ),
}

# table configs
TABLE_CONFIGS = {
    "table": TableConfig(
        name="table",
        p=[0, 0, 0],
        q=[1, 0, 0, 0],
        offset=[0, 0, 0],
        downsample=1,
    ),
}

# camera configs
CAMERA_INTRINSICS_DICT = {
    "image_height": 480,
    "image_width": 640,
    "fx": 604.152161,
    "fy": 604.267212,
    "cx": 316.670074,
    "cy": 243.194473,
}

CAMERA_EXTRINSICS = torch.tensor([
    [0.68370109, -0.50239316, 0.52929587, 0.07500001],
    [0.33318036, 0.86018832, 0.38609201, -0.16100000],
    [-0.64926404, -0.08762054, 0.75549905, 0.49700000],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
])


def get_camera_config(name,camera_int_dict,camera_ext):
    # camera intrinsic
    intrinsic = np.array([
        [camera_int_dict["fx"], 0.0, camera_int_dict["cx"]],
        [0.0, camera_int_dict["fy"], camera_int_dict["cy"]],
        [0.0, 0.0, 1.0]
    ])
    # camera extrinsic
    p = camera_ext[:3, 3]
    q = matrix_to_quaternion(camera_ext[:3, :3])
    pose = sapien.Pose(p=p, q=q)
    # camera config
    return CameraConfig(
        uid=name,
        pose=pose,
        width=camera_int_dict["image_width"],
        height=camera_int_dict["image_height"],
        intrinsic=intrinsic,
        near=0.01,
        far=100,
    )

# default camera configs
CAMERA_CONFIGS = {
    # from bingwen
    "3rd_view_camera": get_camera_config(
        name="3rd_view_camera",
        camera_int_dict=CAMERA_INTRINSICS_DICT,
        camera_ext=CAMERA_EXTRINSICS,
    ),
}   

# GS render configs
GS_RENDER_CONFIG = {
    "back_ground_file": ASSETS_DIR / "background" / "background_image" / "green_background.jpg",
}


