import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData
from dataclasses import dataclass
from typing import List
from typing import Optional
from mani_skill_gs.utils.utils_gs import *

@dataclass
class GaussParams:
    """
    Gaussian Splatting parameters.
    
    Attributes:
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        quats: Gaussian rotations (quaternions) [N, 4]
        features_dc: DC (direct current) SH coefficients [N, 3]
        features_rest: Rest SH coefficients [N, 15, 3]
        opacities: Gaussian opacities [N, 1]
        indices: Segmentation indices of the Gaussians [N, 1]
    """
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacities: torch.Tensor
    indices: torch.Tensor
    
    @property
    def num_points(self) -> int:
        """Number of Gaussians."""
        return self.means.shape[0]
    
    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self.means.device
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format (for backward compatibility).
        
        Returns:
            Dictionary containing all Gaussian parameters with keys:
            'means', 'scales', 'quats', 'features_dc', 'features_rest',
            'opacities', 'indices'
        """
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "features_dc": self.features_dc,
            "features_rest": self.features_rest,
            "opacities": self.opacities,
            "indices": self.indices,
        }
    
    @classmethod
    def from_ply(cls, ply_path: str, device: torch.device, downsample: int = 1) -> 'GaussParams':
        """
        Load GS params from a ply file exported from ns-export.
        
        Args:
            ply_path: Path to the PLY file
            device: Device to load tensors on
            downsample: Optional downsample factor. If provided, randomly samples 
                       every Nth point (e.g., downsample=2 keeps every 2nd point)
            
        Returns:
            GaussParams object with loaded parameters
            
        Example:
            >>> device = torch.device("cuda:0")
            >>> params = GaussParams.from_ply("robot.ply", device)
            >>> print(f"Loaded {params.num_points} Gaussians")
            >>> params_downsampled = GaussParams.from_ply("robot.ply", device, downsample=2)
        """
        ply_path = Path(ply_path)
        print(f"Loading GS params from {ply_path}...")
        plydata = PlyData.read(str(ply_path))

        # xyz
        xyz = np.stack([
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ], axis=1)  # [N, 3]
        num_points = xyz.shape[0]
        print(f"Found {num_points} Gaussians in GS file.")
        
        # opacity
        opacities = np.asarray(plydata.elements[0]["opacity"])  # [N]
        opacities = opacities.reshape(-1, 1)  # [N, 1]
        
        # features_dc
        features_dc = np.zeros((num_points, 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        # features_rest
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        features_rest = np.zeros((num_points, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_rest = features_rest.reshape(num_points, -1, 3)  # [N, 15, 3]
        
        # scale
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((num_points, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # rotation
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        quats = np.zeros((num_points, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            quats[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Apply downsample if specified
        if downsample > 1:
            # Randomly sample points
            original_num_points = num_points
            num_sampled = max(1, num_points // downsample)
            sampled_indices = np.random.choice(num_points, size=num_sampled, replace=False)
            sampled_indices = np.sort(sampled_indices)  # Keep sorted for consistency
            xyz = xyz[sampled_indices]
            opacities = opacities[sampled_indices]
            features_dc = features_dc[sampled_indices]
            features_rest = features_rest[sampled_indices]
            scales = scales[sampled_indices]
            quats = quats[sampled_indices]
            num_points = len(sampled_indices)
            print(f"Downsampled to {num_points} Gaussians (factor: {downsample}, original: {original_num_points})")
        
        return cls(
            means=torch.tensor(xyz, dtype=torch.float32, device=device),
            scales=torch.tensor(scales, dtype=torch.float32, device=device),
            quats=torch.tensor(quats, dtype=torch.float32, device=device),
            features_dc=torch.tensor(features_dc, dtype=torch.float32, device=device),
            features_rest=torch.tensor(features_rest, dtype=torch.float32, device=device),
            opacities=torch.tensor(opacities, dtype=torch.float32, device=device),
            indices=torch.zeros((num_points, 1), dtype=torch.long, device=device),
        )
    
    @classmethod
    def merge(cls, params_list: List['GaussParams']) -> 'GaussParams':
        """
        Merge multiple GaussParams into a single GaussParams.
        
        Args:
            params_list: List of GaussParams objects to merge
            
        Returns:
            Merged GaussParams object
            
        Example:
            >>> robot = GaussParams.from_ply("robot.ply", device)
            >>> table = GaussParams.from_ply("table.ply", device)
            >>> scene = GaussParams.merge([robot, table])
        """
        if not params_list:
            raise ValueError("Cannot merge empty list of params")
        
        if len(params_list) == 0:
            return None
        elif len(params_list) == 1:
            return params_list[0]
        
        return cls(
            means=torch.cat([p.means for p in params_list], dim=0),
            scales=torch.cat([p.scales for p in params_list], dim=0),
            quats=torch.cat([p.quats for p in params_list], dim=0),
            features_dc=torch.cat([p.features_dc for p in params_list], dim=0),
            features_rest=torch.cat([p.features_rest for p in params_list], dim=0),
            opacities=torch.cat([p.opacities for p in params_list], dim=0),
            indices=torch.cat([p.indices for p in params_list], dim=0),
        )

    def add_indices(self, index: int):
        """
        Add indices to the GaussParams for segmentation.
        
        Assigns a unique index to all Gaussians in this object for tracking
        and transformation purposes. This allows multiple objects to be merged
        while maintaining the ability to transform them individually.
        
        Args:
            index: Integer index to assign to all Gaussians
            
        Example:
            >>> robot_params = GaussParams.from_ply("robot.ply", device)
            >>> robot_params.add_indices(0)  # Assign index 0 to robot
            >>> table_params = GaussParams.from_ply("table.ply", device)
            >>> table_params.add_indices(1)  # Assign index 1 to table
        """
        self.indices = torch.tensor([index] * self.means.shape[0], device=self.device)

    def translate(self, translation: torch.Tensor, index: int = -1):
        """
        Apply translation to the Gaussians.
        
        Args:
            translation: Translation vector of shape (3,)
            index: Index of the object to translate. If -1, translates all Gaussians.
            
        Example:
            >>> params = GaussParams.from_ply("object.ply", device)
            >>> offset = torch.tensor([0.1, 0.2, 0.0], device=device)
            >>> params.translate(offset, index=0)  # Translate object with index 0
        """
        mask = self._get_mask(index)
        translation = translation.unsqueeze(0)
        translation = translation.to(self.device)
        self.means[mask] += translation
    
    def rotate(self, quaternions: torch.Tensor, index: int = -1):
        """
        Apply rotation to the Gaussians.
        
        Args:
            quaternions: Rotation quaternion of shape (4,), in [w, x, y, z] format
            index: Index of the object to rotate. If -1, rotates all Gaussians.
            
        Note:
            Also rotates the spherical harmonics (SH) features to maintain visual consistency.
            
        Example:
            >>> params = GaussParams.from_ply("object.ply", device)
            >>> quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # No rotation
            >>> params.rotate(quat, index=0)  # Rotate object with index 0
        """
        mask = self._get_mask(index)
        quaternions = quaternions.to(self.device)
        rotation_matrix = quaternion_to_matrix(quaternions)
        self.means[mask] = torch.matmul(self.means[mask], rotation_matrix.T)
        self.quats[mask] = torch.nn.functional.normalize(quat_multiply(
            self.quats[mask],
            quaternions,
        ))
        self.features_rest[mask] = transform_shs(self.features_rest[mask], rotation_matrix)

    def scale(self, scale_factor: torch.Tensor, index: int = -1):
        """
        Apply scaling to the Gaussians around their center.
        
        Args:
            scale_factor: Scale factor of shape (3,) for x, y, z axes
            index: Index of the object to scale. If -1, scales all Gaussians.
            
        Example:
            >>> params = GaussParams.from_ply("object.ply", device)
            >>> scale = torch.tensor([0.9, 0.9, 0.9], device=device)
            >>> params.scale(scale, index=0)  # Scale object with index 0
        """
        mask = self._get_mask(index)
        scale_factor = scale_factor.to(self.device)
        center = self.means[mask].mean(dim=0, keepdim=True)
        self.means[mask] = center + (self.means[mask] - center) * scale_factor

    def _get_mask(self, index: int = -1):
        if index != -1:
            index = torch.tensor([index], device=self.device)
            mask = self.indices == index
        else:
            mask = torch.ones_like(self.indices, dtype=torch.bool)
        return mask

    def get_center(self) -> torch.Tensor:
        """
        Get the center of the Gaussians.
        
        Returns:
            Center point of shape (3,) computed as the mean of all Gaussian means.
            
        Example:
            >>> params = GaussParams.from_ply("object.ply", device)
            >>> center = params.get_center()
            >>> print(f"Object center: {center}")
        """
        return self.means.mean(dim=0,keepdim=False)

    def __getitem__(self, key):
        """Allow dict-like access: params['means']"""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in GaussParams")

    def __setitem__(self, key, value):
        """Allow dict-like assignment: params['means'] = x"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' is not a valid attribute of GaussParams")

    def keys(self):
        return self.__dict__.keys()

if __name__ == "__main__":
    params = GaussParams()
    print(params.keys())

