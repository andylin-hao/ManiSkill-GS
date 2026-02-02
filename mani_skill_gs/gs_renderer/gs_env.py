from urdfpy import URDF
import numpy as np
import os
from mani_skill_gs.gs_renderer.gs_params import GaussParams
from mani_skill_gs.utils.utils_gs import *
from mani_skill_gs.gs_renderer.gs_model import GaussModel
import gymnasium as gym 
import torch
import pytorch_kinematics as pk
from nerfstudio.cameras.cameras import Cameras, CameraType
from mani_skill_gs.env_simulation.ms_robot import MyFranka
from mani_skill_gs.gs_renderer.gs_viewer import GaussViewer
import threading
import cv2
from mani_skill_gs.env_simulation.ms_configs import FR3_DEFAULT_CFG
from typing import Any, Union, Dict, Optional
from collections import defaultdict
from mani_skill_gs.env_simulation.ms_configs import GS_RENDER_CONFIG

if not hasattr(np, 'float'):
    np.float = np.float64
    np.int = np.int_
    np.complex = np.complex128
    np.bool = np.bool_

class GSRenderWrapper(gym.Wrapper):
    """
    Wrapper for ManiSkill environments that replaces rendering with Gaussian Splatting.
    All physics and simulation remain unchanged, only the render() method is overridden.
    """
    def __init__(
        self, 
        env: gym.Env, 
        render_interface: str = "env", # env,gs,env+gs,gs_rlinf
        back_ground_file: str = GS_RENDER_CONFIG["back_ground_file"],
        cache_table: bool = True,
        render_bs: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Initialize GS renderer wrapper for ManiSkill environment.
        
        Args:
            env: Base ManiSkill environment to wrap
            device: Device to run GS rendering on ("cuda" or "cpu")
            render_interface: Rendering mode - "env" (original), "gs" (GS only),
                or "env+gs" (both)
            back_ground_file: Path to background image file (optional)
            
        Example:
            >>> env = gym.make("MySimpleEnv", robot_uids="my_franka")
            >>> gs_env = GSRenderWrapper(env, device="cuda", render_interface="gs")
        """
        super().__init__(env)
        # ------------------ MS Env ------------------
        self.device = device
        self.num_envs = env.unwrapped.num_envs
        self.actor_dict = env.unwrapped.get_actor_dict()
        self.render_interface = render_interface
        self.back_ground_img = self._prepare_back_ground(back_ground_file)

        ## ----------------- GS Build -----------------
        # params
        self.fg_asset_idx = 0  
        self.bg_asset_idx = 0  
        self.robot_links = []
        self.cache_table = cache_table
        self.cached_bg_img = None
        self.cached_bg_alpha = None
        self.render_bs = render_bs
        # build models
        self.robot_urdf = self._build_robot_urdf()
        self.view_camera = self._build_view_camera()
        fg_params, bg_params = self._build_gs_assets()
        # Foreground model (robot + objects)
        self.foreground_model = GaussModel(
            fg_params, 
            self.view_camera, 
            self.num_envs, 
            self.render_bs,
            self.device
        ) 
        # Background model (table)
        self.background_model = GaussModel(
            bg_params, 
            self.view_camera, 
            self.num_envs, 
            self.render_bs,
            self.device
        )
        # cache table 
        if self.cache_table:
            bg_img_all, bg_alpha_all = self.background_model.render()
            self.cached_bg_img = bg_img_all[0:1].detach().clone() 
            self.cached_bg_alpha = bg_alpha_all[0:1].detach().clone()
            del bg_img_all, bg_alpha_all
            del self.background_model
            torch.cuda.empty_cache()

    ## ----------------- Render -----------------
    @torch.no_grad()
    def render(self) -> dict:
        """
        Override render method to return both GS rendering and original env rendering.
        """
        # Render using GS
        gs_img = None
        env_img = None
        if "gs" in self.render_interface:
            # render
            if self.cache_table:
                bg_img, bg_alpha = self.cached_bg_img, self.cached_bg_alpha
            else:
                bg_img, bg_alpha = self.background_model.render()
            fg_img, fg_alpha = self.foreground_model.render()
            gs_img = bg_img * (1 - fg_alpha) + fg_img * fg_alpha
            alpha = bg_alpha + fg_alpha * (1 - bg_alpha)
            if self.back_ground_img is not None:
                gs_img = self.back_ground_img * (1 - alpha) + gs_img * alpha
            gs_img = (gs_img * 255.0).clamp(0, 255).cpu().to(torch.uint8)
            # clear cuda cache
            del bg_img, bg_alpha, fg_img, fg_alpha, alpha
        # Render using MS Env
        if "env" in self.render_interface:
            env_img = self.env.unwrapped.render()
            env_img = env_img.detach().cpu().to(torch.uint8)
        # Return images
        if self.render_interface == "gs_rlinf":
            return gs_img
        else:
            return {
                'env': env_img,
                'gs': gs_img,
            }

    def _prepare_back_ground(self, back_ground_file):
        """
        Prepare the background image for the environment (For static scene).
        
        Loads and prepares a background image to composite with GS rendering.
        The image is repeated for all parallel environments.
        
        Args:
            back_ground_file: Path to background image file, or None to skip
            
        Returns:
            Background image array of shape [num_envs, H, W, 3], or None
        """
        if back_ground_file is None:
            return None
        back_ground_img = cv2.imread(back_ground_file)
        back_ground_img = cv2.cvtColor(back_ground_img, cv2.COLOR_BGR2RGB)
        back_ground_img = back_ground_img[None]
        back_ground_img = torch.from_numpy(back_ground_img).float().to(self.device) / 255.0
        return back_ground_img

    ## ----------------- Build GS Assets -----------------
    def _build_gs_assets(self):
        """
        Build Gaussian Splatting assets from environment actors.
        
        Loads GS parameters for all actors in the scene (robot, table, objects)
        and separates them into foreground (robot + objects) and background (table).
        
        Returns:
            Tuple of (foreground_params, background_params):
                - foreground_params: Merged GaussParams for robot and objects, or None
                - background_params: GaussParams for table, or None
                
        Note:
            Objects are loaded in sequence and assigned unique indices for
            segmentation and transformation tracking.
        """
        # ! Load in sequence
        fg_assets, bg_assets = [], []
        for actor_name, actor_info in self.actor_dict.items():
            if actor_name == "robot":
                fg_assets.append(self.__build_robot_gs(actor_info))
            elif actor_name == "table":
                bg_assets.append(self.__build_table_gs(actor_info))
            elif "obj" in actor_name:
                fg_assets.append(self.__build_object_gs(actor_info))
        fg_params = GaussParams.merge(fg_assets) 
        bg_params = GaussParams.merge(bg_assets) 
        return fg_params, bg_params

    def __build_robot_gs(self, actor_info):
        """
        Build GS parameters for the robot from multiple link files.
        
        Loads GS parameters for each robot link from separate PLY files,
        transforms them to the correct pose based on URDF forward kinematics,
        and merges them into a single GaussParams object.
        
        Args:
            gs_path_dir: Directory containing robot link GS files
            urdf_path: Path to robot URDF file
            offset: Additional offset to apply to robot pose [3,]
            
        Returns:
            Merged GaussParams for all robot links
            
        Note:
            Some links (fr3_link8, hand_tcp, sensors, pads, camera) are skipped
            as they don't have GS representations or are not visible.
        """
        ## build gs model
        # urdf assets
        robot = URDF.load(actor_info["urdf_path"])
        # gs assets
        fk = robot.link_fk(FR3_DEFAULT_CFG)
        robot_gs_list = []
        self.robot_links = []
        # transmat denote the link-to-base transform
        for _, (urdf_link, transmat) in enumerate(fk.items()):
            part_name = urdf_link.name
            if part_name == "fr3_link8":
                continue
            if part_name == "fr3_hand_tcp" or "sc" in part_name or "pad" in part_name or "camera" in part_name:
                continue
            # read gs params
            gs_params = GaussParams.from_ply(
                os.path.join(actor_info["gs_path"], f"{part_name}_3dgs_abs.ply"), 
                device=self.device, 
                downsample=actor_info["downsample"],
            )
            gs_params.add_indices(self.fg_asset_idx)
            self.robot_links.append(part_name)
            # transform gs params from base link to target link
            trans = -torch.from_numpy(transmat[:3, 3]).float()
            rot = torch.from_numpy(np.linalg.inv(transmat[:3, :3])).float()
            quat = matrix_to_quaternion(rot)
            gs_params.translate(trans)
            gs_params.translate(actor_info["offset"])
            gs_params.rotate(quat)
            robot_gs_list.append(gs_params)
            self.fg_asset_idx += 1
        robot_gs = GaussParams.merge(robot_gs_list)
        return robot_gs

    def __build_table_gs(self, actor_info):
        """
        Build GS parameters for the table.
        
        Args:
            gs_path: Path to table GS PLY file
            
        Returns:
            GaussParams for the table with assigned index
        """
        table_gs = GaussParams.from_ply(
            actor_info["gs_path"], 
            device=self.device, 
            downsample=actor_info["downsample"],
        )
        table_gs.add_indices(self.bg_asset_idx)
        table_gs.translate(actor_info["offset"])
        self.bg_asset_idx += 1
        return table_gs

    def __build_object_gs(self, actor_info):
        """
        Build GS parameters for a scene object.
        
        Loads object GS parameters, centers them at origin, then applies
        offset and scale transformations.
        
        Args:
            actor_info: Actor information dictionary
            
        Returns:
            GaussParams for the object with transformations applied
        """
        object_gs = GaussParams.from_ply(
            actor_info["gs_path"], 
            device=self.device, 
            downsample=actor_info["downsample"],
        )
        object_gs.add_indices(self.fg_asset_idx)
        object_gs.translate(-object_gs.get_center())
        object_gs.translate(actor_info["offset"])
        object_gs.scale(actor_info["gs_scale"])
        self.fg_asset_idx += 1
        return object_gs

    ## ----------------- Build URDF Model -----------------
    def _build_robot_urdf(self):
        """
        Build the URDF model for GS params update.
        
        Creates a kinematic chain from the robot URDF file for computing
        forward kinematics. This is used to update robot link poses during
        simulation.
        
        Returns:
            Kinematic chain object for forward kinematics computation
            
        Note:
            The kinematic chain is built using pytorch-kinematics and placed
            on the specified device for efficient computation.
        """
        urdf_path = self.actor_dict["robot"]["urdf_path"]
        robot = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read()).to(device=self.device)
        return robot

    ## ----------------- Build View Camera -----------------
    def _build_view_camera(self):
        """
        Build the camera with an angled view for GS rendering.
        
        Creates a camera configuration that matches the ManiSkill environment's
        camera setup. Uses a look-at matrix to position the camera.
        
        Returns:
            Cameras object configured for GS rendering
            
        Note:
            Camera intrinsics: fx=400.0, fy=400.0, cx=320.0, cy=240.0
            Resolution: 640x480
        """
        # TODO: support the wrist camera
        config = self.env.unwrapped._default_sensor_configs[0]
        # camera axis conversion
        pose = config.pose
        R = quaternion_to_matrix(pose.q[:])[0].to(self.device)  # [3, 3]
        t = pose.p.to(self.device)[0][:, None]  # [3, 1]
        c2w = torch.cat([-R[:, 1:2], R[:, 2:3], -R[:, 0:1], t], dim=1).unsqueeze(0)  # [1, 3, 4]
        # camera dict
        camera_dict = {}
        camera_dict["camera_to_worlds"] = c2w
        camera_dict["fx"] = config.intrinsic[0, 0]
        camera_dict["fy"] = config.intrinsic[1, 1]
        camera_dict["cx"] = config.intrinsic[0, 2]
        camera_dict["cy"] = config.intrinsic[1, 2]
        camera_dict["width"] = config.width
        camera_dict["height"] = config.height
        camera_dict["camera_type"] = CameraType.PERSPECTIVE
        # Build camera object
        camera = Cameras(**camera_dict).to(self.device)
        return camera

    ## ----------------- Update GS Assets -----------------
    def update_gs_params(self):
        """
        Update GS parameters based on current environment state.
        
        Computes transformation matrices for all objects (robot links, table, objects)
        based on their current poses in the simulation, and applies these transformations
        to the GS parameters. This must be called after each env.step() to keep
        GS rendering synchronized with the simulation.
        
        Note:
            For the robot, uses forward kinematics to compute link poses from joint angles.
            For static objects (table) and dynamic objects, uses their current pose.
            If cache_table is True, table transformations are skipped entirely.
            
        Example:
            >>> obs, reward, done, truncated, info = env.step(action)
            >>> env.update_gs_params()  # Update GS rendering
            >>> render_result = env.render()
        """
        fg_mats, bg_mats = [], []
        for actor_name, actor_info in self.actor_dict.items():
            if actor_name == "robot":
                robot = actor_info["actor"]
                qpos = robot.get_qpos().to(self.device)
                ret = self.robot_urdf.forward_kinematics(qpos)
                for link_name in self.robot_links:
                    link_trans = ret[link_name].get_matrix()
                    fg_mats.append(link_trans)
                del qpos, ret
            elif "obj" in actor_name:
                pose = actor_info["actor"].pose
                mat = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
                mat[:, :3, :3] = quaternion_to_matrix(pose.q)
                mat[:, :3, 3] = pose.p
                fg_mats.append(mat)
            elif actor_name == "table":
                if not self.cache_table:
                    pose = actor_info["actor"].pose
                    mat = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
                    mat[:, :3, :3] = quaternion_to_matrix(pose.q)
                    mat[:, :3, 3] = pose.p
                    bg_mats.append(mat)
        # Update foreground model
        fg_all_mats = torch.stack(fg_mats, dim=1)
        fg_transmat_list = list(fg_all_mats.reshape(-1, 4, 4).split(1, dim=0))
        self.foreground_model.transform_gauss_params(fg_transmat_list)
        del fg_mats, fg_all_mats, fg_transmat_list
        # Update background model 
        if not self.cache_table:
            bg_all_mats = torch.stack(bg_mats, dim=1)
            bg_transmat_list = list(bg_all_mats.reshape(-1, 4, 4).split(1, dim=0))
            self.background_model.transform_gauss_params(bg_transmat_list)
            del bg_mats, bg_all_mats, bg_transmat_list

    ## ----------------- Open Viewer -----------------
    def open_viewer(self):
        """
        Open viewer in a non-blocking thread.
        
        Launches an interactive Nerfstudio viewer in a separate daemon thread,
        allowing the main process to continue controlling the environment.
        GS parameters will be automatically updated when render() is called.
        
        Note:
            The viewer runs in a daemon thread, so it will automatically stop
            when the main process exits. Use Ctrl+C in the viewer to stop it.
            
        Example:
            >>> env = GSRenderWrapper(base_env, device="cuda")
            >>> env.open_viewer()  # Opens viewer in background
            >>> # Continue using environment normally
            >>> obs, info = env.reset()
        """
        # viewer loop in separate thread to avoid blocking
        def viewer_loop():
            viewer = GaussViewer(self.foreground_model)
            viewer.open()
        viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
        viewer_thread.start()

    def get_language_instruction(self):
        """
        Get the language instruction for the environment.
        
        Returns:
            Language instruction for the environment
        """
        return self.env.unwrapped.get_language_instruction()

    # --------------- align with the obs interface in RLinf ---------------
    def get_obs_dict(self):
        sensor_data = defaultdict[Any, dict](dict)
        if self.render_interface == "gs_rlinf":
            sensor_data["3rd_view_camera"]["rgb"] = self.render()
        else:
            output_dict = self.render()
            sensor_data["3rd_view_camera"]["gs"] = output_dict['gs']
            sensor_data["3rd_view_camera"]["env"] = output_dict['env']
        # obs dict
        obs_dict = self.env.unwrapped.get_init_obs_dict() # get agent state from ms env, which is not accessible in wrapper
        obs_dict["sensor_data"] = sensor_data
        return obs_dict

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        with torch.no_grad():
            _, reward, terminated, truncations, info = super().step(action=action)
            if isinstance(truncations, bool):
                truncations = torch.tensor([truncations], device=self.device)
                truncations = truncations.repeat(terminated.shape[0])
            self.update_gs_params()
            obs = self.get_obs_dict()
        return obs, reward, terminated, truncations, info

    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        with torch.no_grad():
            _, info = super().reset(seed=seed, options=options)
            self.update_gs_params()
            obs = self.get_obs_dict()
        return obs, info
    
        
        
        
