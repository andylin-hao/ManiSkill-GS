import torch
import torch.nn as nn
from typing import Optional
import os
import sys
import math
cuda_bin = "/usr/local/cuda-12.4/bin"
current_path = os.environ.get("PATH", "")
if cuda_bin not in current_path:
    os.environ["PATH"] = f"{cuda_bin}:{current_path}" if current_path else cuda_bin

from nerfstudio.models.splatfacto import SplatfactoModelConfig
from mani_skill_gs.gs_renderer.splat_facto_model import SplatfactoModel_v2
from nerfstudio.cameras.cameras import Cameras
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_multiply
from typing import List
from mani_skill_gs.gs_renderer.gs_params import GaussParams
import copy
import numpy as np
from nerfstudio.data.scene_box import SceneBox
import copy

# TODO: support multiple cameras like 3rd camera and wrist camera for parallel rendering
class GaussModel:
    """
    Gaussian Splatting model wrapper.
    
    This class encapsulates a SplatfactoModel and provides convenient methods
    for loading parameters and rendering.
    """
    
    def __init__(
        self, 
        gauss_params: GaussParams, 
        camera: Cameras, 
        num_env: int, 
        render_bs: Optional[int],
        device: torch.device,
    ):
        """
        Initialize a Gaussian Splatting model.
        
        Args:
            gauss_params: Initial Gaussian parameters
            camera: Camera configuration for rendering
            num_env: Number of parallel environments for vectorized rendering
            device: Device to run the model on (cuda/cpu)
            
        Example:
            >>> params = GaussParams.from_ply("scene.ply", device)
            >>> camera = Cameras(...)
            >>> model = GaussModel(params, camera, num_env=1, device=device)
        """
        self.device = device
        self.num_env = num_env
        self.model = self._build_gs_model(device)
        self.gauss_params, self.camera = self._build_vector_env(
            gauss_params,
            camera,
        )
        self.render_bs = self.num_env if render_bs is None else render_bs
        self._load_params_to_model(self.gauss_params)

    def _build_vector_env(self, gauss_params: GaussParams, camera: Cameras):
        """
        Build a vector 3DGS environment for parallel rendering.
        
        Creates multiple copies of the scene arranged in a grid for efficient
        parallel rendering of multiple environments. Each environment is offset
        spatially to avoid overlap.
        
        Args:
            gauss_params: Base Gaussian parameters to replicate
            camera: Base camera configuration to replicate
            
        Returns:
            Tuple of (vector_gauss_params, vector_camera) for parallel rendering
            
        Note:
            The grid layout is sqrt(num_env) x sqrt(num_env). Each environment
            is offset by 5 units in the grid.
        """
        # params
        grid_size = math.ceil(math.sqrt(self.num_env))
        self.index_offset = gauss_params.indices.max() + 1
        offset_scale = 5
        # vector gauss
        vector_gauss_dict = {}
        offset_list = []
        for name in gauss_params.keys():
            param = gauss_params[name]
            param_list = []
            for idx in range(self.num_env):
                if name == "means":
                    row = idx // grid_size
                    col = idx % grid_size
                    current_offset = torch.tensor([col * offset_scale, row * offset_scale, 0], device=self.device)
                    param_idx = param + current_offset
                    offset_repeated = current_offset.unsqueeze(0).repeat(param.shape[0], 1)
                    offset_list.append(offset_repeated)
                elif name == "indices":
                    param_idx = param + idx * self.index_offset
                else:
                    param_idx = param
                param_list.append(param_idx)
            params = torch.cat(param_list, dim=0)
            vector_gauss_dict[name] = params
        vector_gauss_params = GaussParams(**vector_gauss_dict)
        self.offsets = torch.cat(offset_list, dim=0) # [Total_Points, 3]
        
        # vector camera
        c2w = camera.camera_to_worlds # [1, 3, 4]
        c2w_vector_list = []
        for idx in range(self.num_env):
            row = idx // grid_size
            col = idx % grid_size
            offset = torch.tensor([col * offset_scale, row * offset_scale, 0], device=self.device)
            c2w_idx = torch.zeros_like(c2w)
            c2w_idx[: , :, :3] = c2w[:, :, :3]
            c2w_idx[:, :, 3] = c2w[:, :, 3] + offset
            c2w_vector_list.append(c2w_idx)
        vector_camera = copy.deepcopy(camera)
        vector_camera.camera_to_worlds = torch.cat(c2w_vector_list, dim=0)
        return vector_gauss_params, vector_camera
            
            
    def _build_gs_model(self, device: torch.device) -> SplatfactoModel_v2:
        """
        Build an empty SplatfactoModel for GS rendering.
        
        Creates a new SplatfactoModel instance with default configuration.
        The model is initialized empty and parameters will be loaded later.
        
        Args:
            device: Device to place the model on
            
        Returns:
            SplatfactoModel instance in eval mode
        """
        print("Initializing empty model...")
        model_config = SplatfactoModelConfig(
            output_depth_during_training = False,
        )
        model = SplatfactoModel_v2(
            config=model_config,
            num_train_data=1,
            scene_box=None,
        )
        model = model.to(device)
        model.eval()
        return model
        
    def _load_params_to_model(self, gauss_params: GaussParams):
        """
        Load GS params into the SplatfactoModel.
        
        Converts GaussParams to the format expected by SplatfactoModel and
        sets them as model parameters. Indices are stored separately for
        segmentation purposes.
        
        Args:
            gauss_params: GaussParams object to load into the model
            
        Note:
            Indices are stored in model.index_params, while other parameters
            are stored in model.gauss_params as nn.Parameter objects.
        """
        params_dict = gauss_params.to_dict()
        self.model.index_params = params_dict["indices"]
        params_dict.pop("indices")
        self.model.gauss_params = nn.ParameterDict({
            k: nn.Parameter(v) for k, v in params_dict.items()
        })
    
    @torch.no_grad()
    def render(self) -> dict:
        """
        Render the model and return RGB image and alpha channel.
        
        Returns:
            Tuple of (img_np, alpha_np):
                - img_np: Rendered RGB image of shape [num_env, H, W, 3], uint8
                - alpha_np: Alpha channel of shape [num_env, H, W, 1], float32
                
        Example:
            >>> model = GaussModel(params, camera, num_env=1, device=device)
            >>> img, alpha = model.render()
            >>> print(f"Rendered image shape: {img.shape}")
        """
        num_cams = self.camera.camera_to_worlds.shape[0]

        # render in one batch
        if num_cams <= self.render_bs:
            outputs = self.model.get_outputs_for_camera(self.camera)
            img = outputs["rgb"]          
            alpha = outputs["accumulation"]
            return img, alpha

        # render in micro batches
        rgb_list = []
        alpha_list = []
        for start in range(0, num_cams, self.render_bs):
            end = min(start + self.render_bs, num_cams)
            sub_camera = copy.deepcopy(self.camera)
            sub_camera.camera_to_worlds = sub_camera.camera_to_worlds[start:end]
            sub_outputs = self.model.get_outputs_for_camera(sub_camera)
            sub_rgb = sub_outputs["rgb"]            
            sub_alpha = sub_outputs["accumulation"]

            if sub_rgb.dim() == 3:
                sub_rgb = sub_rgb.unsqueeze(0)
                sub_alpha = sub_alpha.unsqueeze(0)

            rgb_list.append(sub_rgb)
            alpha_list.append(sub_alpha)

        img = torch.cat(rgb_list, dim=0)      
        alpha = torch.cat(alpha_list, dim=0) 
        return img, alpha

    def get_params(self) -> GaussParams:
        """
        Get the currently loaded GaussParams from the model.
        
        Extracts the current Gaussian parameters from the model and returns
        them as a GaussParams object. Useful for inspection or saving.
        
        Returns:
            GaussParams object containing current model parameters
            
        Example:
            >>> model = GaussModel(params, camera, num_env=1, device=device)
            >>> current_params = model.get_params()
            >>> print(f"Model has {current_params.num_points} Gaussians")
        """
        return GaussParams(
            means=self.model.gauss_params["means"],
            scales=self.model.gauss_params["scales"],
            quats=self.model.gauss_params["quats"],
            features_dc=self.model.gauss_params["features_dc"],
            features_rest=self.model.gauss_params["features_rest"],
            opacities=self.model.gauss_params["opacities"],
            indices=self.model.index_params
        )
    
    @property
    def num_points(self) -> int:
        """
        Number of Gaussians in the model.
        """
        return self.model.num_points

    def transform_gauss_params(self, transmat_list: List[torch.Tensor], fast_mode: bool = True):
        """
        Transform the Gauss params by a transformation matrix.
        
        Updates the Gaussian parameters based on the current pose of objects
        in the scene. This is called after each environment step to synchronize
        GS rendering with the simulation state.
        
        Args:
            transmat_list: List of transformation matrices, one per object.
                Each matrix is of shape [num_env, 4, 4] or [1, 4, 4]
            fast_mode: If True, uses optimized batch transformation. If False,
                uses per-object transformation (slower but more flexible).
                
        Note:
            In fast_mode, all transformations are applied in a single batch operation
            for efficiency. The order of objects in transmat_list must match the
            order of indices in gauss_params.
            
        Example:
            >>> # After env.step(), update GS params
            >>> robot_mat = get_robot_transformation_matrix(env)
            >>> table_mat = get_table_transformation_matrix(env)
            >>> model.transform_gauss_params([robot_mat, table_mat])
        """
        if fast_mode:
            # transform
            transforms = torch.cat(transmat_list,dim = 0) # Shape: (Num_Objects, 4, 4)
            obj_indices = self.gauss_params.indices
            point_transforms = transforms[obj_indices] # transform matrix for each gs point
            R = point_transforms[:, :3, :3] # (N_points, 3, 3)
            T = point_transforms[:, :3, 3]  # (N_points, 3)
            original_means = self.gauss_params.means - self.offsets
            transformed_means = torch.bmm(R, original_means.unsqueeze(-1)).squeeze(-1) + T
            transformed_means = transformed_means + self.offsets
            rot_quats = matrix_to_quaternion(R) # (N_points, 4)
            original_quats = self.gauss_params.quats
            transformed_quats = quaternion_multiply(rot_quats, original_quats)
            # new params
            self.model.gauss_params["means"].data = transformed_means
            self.model.gauss_params["quats"].data = transformed_quats
            del transforms, point_transforms, R, T, original_means, transformed_means, rot_quats, original_quats, transformed_quats
        else:
            self.render_params = copy.deepcopy(self.gauss_params)
            for idx, transmat in enumerate(transmat_list):
                translation = transmat[0, :3, 3]
                rotation = transmat[0, :3, :3]
                quat = matrix_to_quaternion(rotation)
                # ! sequence is important
                self.render_params.rotate(quat, idx)
                self.render_params.translate(translation, idx)
            self._load_params_to_model(self.render_params)

