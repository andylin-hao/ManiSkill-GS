import torch
import sapien
import numpy as np
from copy import deepcopy
from mani_skill.agents.controllers import *
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.robots.panda.panda_wristcam import Panda, PandaWristCam
from mani_skill.agents.registration import register_agent
from mani_skill_gs.env_simulation.ms_configs import ROBOT_CONFIGS, FR3_DEFAULT_CFG


@register_agent()
class MyFranka(BaseAgent):
    uid = "my_franka"
    urdf_path = ROBOT_CONFIGS["my_franka"].urdf_path

    tmp_joint_names = list(FR3_DEFAULT_CFG.keys())
    arm_joint_names = tmp_joint_names[:7]
    gripper_joint_names = tmp_joint_names[7:]
    ee_link_name = "fr3_hand_tcp"
    finger_link_names = ["fr3_leftfinger", "fr3_rightfinger"]
    finger_pad_link_names = ["fr3_leftfinger_pad", "fr3_rightfinger_pad"]
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            finger_link_names[0]:dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            finger_link_names[1]:dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    # arm_stiffness = [
    #     198.0,
    #     95.06666666666666,
    #     183.39595959595957,
    #     112.6222222222222,
    #     2.781818181818179,
    #     8.91212121212121,
    #     436.99999999999994,
    # ]   
    # arm_damping = [
    #     55.0,
    #     20.666666666666668,
    #     39.868686868686865,
    #     20.11111111111111,
    #     0.545454545454545,
    #     1.7474747474747474,
    #     95.0
    # ]
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            normalize_action=False,
        )
        
        arm_pd_joint_target_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            use_target=True,
            normalize_action=False,
        )
        
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
            normalize_action=False,
        )
        
        arm_pd_ee_target_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
            use_target=True,
            normalize_action=False,
        )
        
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos,
                balance_passive_force=False
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos,
                balance_passive_force=False
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos,
                balance_passive_force=False
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos,
                balance_passive_force=False
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos,
                balance_passive_force=False
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy(controller_configs)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.finger_link_names[0]
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.finger_link_names[1]
        )
        self.finger1pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.finger_pad_link_names[0]
        )
        self.finger2pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.finger_pad_link_names[1]
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
