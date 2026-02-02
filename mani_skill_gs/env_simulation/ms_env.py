import sapien
import torch
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill_gs.env_simulation.ms_configs import *
from mani_skill_gs.env_simulation.ms_robot import * 
from typing import Union, Dict
from collections import defaultdict
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from sapien.physx import PhysxMaterial
from mani_skill.utils.structs import Actor, Link
from sapien.physx import PhysxMaterial
from sapien.physx import PhysxRigidBodyComponent

# TODO: flexible interface for multiple environments
# ref: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/minimal_template.py
# 1. load agent
# 2. load scene
# 3. sensor configs
# 4. initialize episode
# TODO: add the fore-ground pre-render technique
@register_env("PutCubeOnPlate-v0", max_episode_steps=100)
class GSEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["my_franka"]
    agent: Union[MyFranka]
    
    def __init__(self, *args, robot_uids, robot_init_qpos_noise=0.02, **kwargs):
        self.actor_dict = {}
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    # TODO: simulation config influence
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=500, # 100
            control_freq=10, # 20
            scene_config=SceneConfig(
                contact_offset=0.005, # 0.02
                solver_position_iterations=25,
                solver_velocity_iterations=2,
            ),
        )

    ## ---------- load agent ----------
    def _load_agent(self, options: dict):
        config = ROBOT_CONFIGS[self.robot_uids]
        super()._load_agent(options, sapien.Pose(p=config.p, q=config.q)) 
        self.actor_dict["robot"] = {
            "actor": self.agent.robot,
            **vars(config)
        }

    ## ---------- load scene ----------
    def _load_scene(self, options: dict):
        # 1. Load table
        self.table = self.__build_table(TABLE_CONFIGS["table"])
        # 2. Load obj
        self.obj_cube = self.__build_obj(OBJECT_CONFIGS["cube"])
        # 3. Load bowl
        self.obj_plate = self.__build_obj(OBJECT_CONFIGS["plate"])

    # TODO: table function sorted
    # load table 
    def __build_table(self, config: TableConfig):
        builder = self.scene.create_actor_builder()
        builder.add_plane_collision(
            pose=sapien.Pose(p=[0, 0, 0.0], q=[0.707, 0, -0.707, 0]),
        )
        builder.set_initial_pose(sapien.Pose(p=config.p, q=config.q))
        builder.add_plane_visual(
            pose=sapien.Pose(p=[0, 0, 0.0], q=[0.707, 0, -0.707, 0]),
            material=[40, 40, 40]  
        )
        table_actor = builder.build_kinematic(name=config.name)
        self.actor_dict["table"] = {
            "actor": table_actor,
            **vars(config)
        }
        return table_actor

    # load object
    def __build_obj(self, config: ObjectConfig):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=config.visual_file, 
            scale=config.mesh_scale,
        )
        builder.add_convex_collision_from_file(
            filename=config.collision_file, 
            scale=config.mesh_scale,
            material=PhysxMaterial(
                static_friction=config.static_friction,
                dynamic_friction=config.dynamic_friction,
                restitution=config.restitution,
            ),
            density=config.density,
            patch_radius=config.patch_radius,
            min_patch_radius=config.min_patch_radius,
        )
        builder.set_initial_pose(sapien.Pose(p=config.p, q=config.q))
        object_actor = builder.build(name=config.name)
        self.actor_dict[config.name] = {
            "actor": object_actor, 
            **vars(config)
        }
        return object_actor

    ## ---------- initialize episode ----------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.__init_scene(env_idx)
            self.__init_objects(env_idx)
            self.__init_robot(env_idx)
    
    # initialize scene
    def __init_scene(self, env_idx):
        pass
        
    # initialize robot
    def __init_robot(self, env_idx):
        b = len(env_idx)   
        config = ROBOT_CONFIGS[self.robot_uids]
        num_joints = len(FR3_DEFAULT_CFG.keys())
        qpos = np.array([[FR3_DEFAULT_CFG[key] for key in FR3_DEFAULT_CFG.keys()] for _ in range(b)])
        qpos = (
            self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, num_joints)
            )
            + qpos
        )
        qpos[:, -2:] = 0.04
        qpos = torch.from_numpy(qpos)
        self.agent.robot.set_pose(sapien.Pose(p=config.p, q=config.q))
        self.agent.reset(qpos)

    # initialize objects
    def __init_objects(self, env_idx):
        max_retries = 3
        for retry in range(max_retries):
            # iterate over all objects
            for obj_name, obj_info in self.actor_dict.items():
                if "object/" not in obj_name:
                    continue
                config = OBJECT_CONFIGS[obj_name.split("/")[-1]]
                # position
                pose = self.__random_perturb_pose(
                    env_idx,
                    p=config.p,
                    q=config.q,
                    p_random=config.p_random,
                )
                obj_info["actor"].set_pose(pose)
                # physics
                for i, obj in enumerate(obj_info["actor"]._objs):
                    if isinstance(obj, Link):
                        obj = obj.entity
                    rigid_body_component: PhysxRigidBodyComponent = obj.find_component_by_type(
                        PhysxRigidBodyComponent
                    )
                    if rigid_body_component is not None:
                        rigid_body_component.mass = config.mass
                        rigid_body_component.cmass_local_pose.set_p(
                            [config.com_x, config.com_y, config.com_z]
                        )
                        for shape in rigid_body_component.collision_shapes:
                            shape.physical_material.dynamic_friction = config.dynamic_friction
                            shape.physical_material.static_friction = config.static_friction
                            shape.physical_material.restitution = config.restitution
                            shape.patch_radius = config.patch_radius
                            shape.min_patch_radius = config.min_patch_radius
                            shape.contact_offset = config.contact_offset
            # TODO: this will cause some bugs on the scenes
            # self._settle()


    # success interface for maniskill
    def evaluate(self):
        return {
            "success": torch.tensor(self.is_success(), dtype=torch.bool, device=self.device),
        }

    # success check for pick-and-place task
    def is_success(self, dist_xy_thresh: float = 0.07, height_margin: float = 0.07) -> bool:
        """
        Check whether the cube is successfully placed on (or very close to) the plate.
        """
        # Positions can be tensors or numpy arrays depending on backend; normalize to np.ndarray
        cube_pos = self.obj_cube.pose.p.cpu().numpy()
        plate_pos = self.obj_plate.pose.p.cpu().numpy()
        # distance
        dist_xy = np.linalg.norm(cube_pos[:, :2] - plate_pos[:, :2], axis=1)
        # cube should be above plate but not too far away in Z
        dz = cube_pos[:, 2] - plate_pos[:, 2]
        height_ok = (dz >= 0.0) & (dz <= height_margin)
        success = (dist_xy < dist_xy_thresh) & height_ok
        return success
    
    # get language instruction
    def get_language_instruction(self) -> str:
        instruct = []
        for idx in range(self.num_envs):
            instruct.append("pick up the cube and put it on the plate")
        return instruct

    # get actor dict
    def get_actor_dict(self):
        return self.actor_dict

    # add random perturbation to pose
    def __random_perturb_pose(self, env_idx, p: torch.Tensor, q: torch.Tensor, p_random: torch.Tensor):
        b = len(env_idx)
        p_base = p.unsqueeze(0)  # [1, 3]
        random_p = p_base + (torch.rand(b, 3, device="cpu") * 2 - 1) * p_random.unsqueeze(0)  # [1, 3]
        return Pose.create_from_pq(p=random_p, q=q)

    # compute dense reward
    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    # compute sparse reward
    def compute_sparse_reward(self, obs, action: torch.Tensor, info: dict):
        success = self.evaluate()["success"]
        return success.float()

    # compute normalized dense reward
    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    # get obs dict
    def get_init_obs_dict(self):
        obs_dict = dict(
            agent=self._get_obs_agent(),
            extra=dict(),
            sensor_param=self.get_sensor_params(),
            sensor_data=dict(),
        )
        return obs_dict

    def _settle(self, t: float = 0.5):
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
        time_steps = max(1, int(t * self.sim_freq / self.control_freq))
        for _ in range(time_steps):
            self.scene.step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    # default sensor configs 
    @property
    def _default_sensor_configs(self):
        return [CAMERA_CONFIGS["3rd_view_camera"]]

    # default human render camera configs
    @property
    def _default_human_render_camera_configs(self):
        return [CAMERA_CONFIGS["3rd_view_camera"]]

