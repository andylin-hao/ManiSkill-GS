import numpy as np
import torch
from PIL import Image
from mani_skill.envs.sapien_env import BaseEnv


class DataCollector:
    def __init__(self, env: BaseEnv):
        """Collect image, state, action and instruction along a trajectory."""
        self.env = env
        self.data_dict = self.get_empty_data_dict()
        self.last_ee_pose = None

    def get_empty_data_dict(self):
        return {
            "image": [],
            "instruction": None,
            "action": [],  # delta EE pose (6) + gripper (1) = 7
            "state": [],   # end effector pose (xyz + euler angles) (6) + gripper (1) = 7
        }

    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()
        self.last_ee_pose = None

    def update_data_dict(self, obs, action):
        """Update all modalities once before env.step()."""
        # language instruction
        self.data_dict["instruction"] = self.env.get_language_instruction()[0] 
        # obs
        rgb = obs["sensor_data"]["3rd_view_camera"]["rgb"][0].numpy()
        self.data_dict["image"].append(Image.fromarray(rgb).convert("RGB"))
        state = obs["agent"]["qpos"][0]
        state = state.numpy()
        self.data_dict["state"].append(state)
        # action
        self.data_dict["action"].append(action.astype(np.float32))

    def save_data(self):
        """ Save data as dictionary. """
        saving_data = {
            "image": self.data_dict["image"],
            "instruction": self.data_dict["instruction"],
            "action": np.array(self.data_dict["action"]), 
            "state": np.array(self.data_dict["state"]),    
        }
        self.clear_data()
        return saving_data

