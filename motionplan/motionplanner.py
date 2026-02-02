import mplib
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

class FR3MotionPlanningSolver(PandaArmMotionPlanningSolver):
    OPEN = 1
    CLOSED = -0.5
    MOVE_GROUP = "fr3_hand_tcp"
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = False,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        data_collector=None,
    ):
        super().__init__(
            env,
            debug=debug,
            vis=False,
            base_pose=base_pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=joint_vel_limits,
            joint_acc_limits=joint_acc_limits,
        )
        self.data_collector = data_collector
        self.last_ee_pose_7d = None  # Stores [x, y, z, w, x, y, z]

    def setup_planner(self):
        """Initialize MPLIB planner with robot SRDF/URDF."""
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        """Execute the planned trajectory."""
        n_step = result["position"].shape[0]
        obs = self.env.get_obs_dict()
        # Follow the planned trajectory
        for i in range(n_step + refine_steps):
            qpos_plan = result["position"][min(i, n_step - 1)]
            delta_ee = self.compute_delta_ee_pose_action(qpos_plan)
            action = np.concatenate([delta_ee, [self.gripper_state]])
            self.data_collector.update_data_dict(obs, action)
            obs, reward, terminated, truncated, info = self.env.step(action[None])
            self.elapsed_steps += 1
            
        return obs, reward, terminated, truncated, info

    def compute_delta_ee_pose_action(self, target_qpos: np.ndarray) -> np.ndarray:
        """
        Compute the 6D delta action (pos_delta + rot_vec) to reach target qpos.
        Correctly handles 3D rotation using Quaternions -> Axis-Angle.
        """
        # Get target 7D pose [pos(3), quat(4)]
        target_ee_pose_7d = self._get_ee_pose_7d_from_qpos(target_qpos)
        # Initialize last_ee_pose
        if self.last_ee_pose_7d is None:
            num_joints = len(self.planner.joint_vel_limits)
            current_qpos_main = self.robot.get_qpos()[0, :num_joints].cpu().numpy()
            self.last_ee_pose_7d = self._get_ee_pose_7d_from_qpos(current_qpos_main)
        # Compute the position delta
        current_pos = self.last_ee_pose_7d[:3]
        target_pos = target_ee_pose_7d[:3]
        delta_pos = target_pos - current_pos
        # Compute the rotation delta
        current_quat_wxyz = self.last_ee_pose_7d[3:]
        target_quat_wxyz = target_ee_pose_7d[3:]
        # Reorder to [x, y, z, w] for Scipy
        current_quat_xyzw = current_quat_wxyz[[1, 2, 3, 0]]
        target_quat_xyzw = target_quat_wxyz[[1, 2, 3, 0]]
        # Compute the rotation delta
        r_current = R.from_quat(current_quat_xyzw)
        r_target = R.from_quat(target_quat_xyzw)
        # Compute the rotation delta
        r_diff = r_target * r_current.inv()
        # Convert to Rotation Vector (Axis-Angle), which pd_ee_delta_pose expects
        delta_rot_vec = r_diff.as_rotvec()
        # Combine into 6D action
        delta_action = np.concatenate([delta_pos, delta_rot_vec])
        # Update the last ee pose
        self.last_ee_pose_7d = target_ee_pose_7d.copy()
        return delta_action

    def _get_ee_pose_7d_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute End-Effector (EE) pose (xyz + quaternion) for a given qpos.
        Returns: np.ndarray of shape (7,) -> [x, y, z, w, x, y, z] (Sapien Quat Convention)
        """
        # Snapshot current state
        current_qpos = self.robot.get_qpos()
        device = self.env.unwrapped.device
        # Get the ee pose from the qpos
        num_joints = len(self.planner.joint_vel_limits) # 7
        current_gripper = current_qpos[0, num_joints:].cpu().numpy()
        full_qpos = np.concatenate([qpos[:num_joints], current_gripper])
        qpos_tensor = torch.tensor(full_qpos, device=device).unsqueeze(0)
        self.robot.set_qpos(qpos_tensor)
        ee_pose = self.env.agent.tcp.pose
        pos = ee_pose.p[0].cpu().numpy()
        quat = ee_pose.q[0].cpu().numpy() # [w, x, y, z]
        # Restore original state
        self.robot.set_qpos(current_qpos)
        return np.concatenate([pos, quat])

    def _move_gripper(self, target_state: int, t: int):
        """Unified internal method to handle gripper open/close logic."""
        # Update the gripper state
        self.gripper_state = target_state
        # Step the environment
        for _ in range(t):
            action = np.concatenate([np.zeros(6), np.array([target_state])])
            obs = self.env.get_obs_dict()
            self.data_collector.update_data_dict(obs, action)
            obs, reward, terminated, truncated, info = self.env.step(action[None])
            self.elapsed_steps += 1
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=20, gripper_state=None):
        return self._move_gripper(self.CLOSED, t)

    def open_gripper(self, t=6, gripper_state=None):
        return self._move_gripper(self.OPEN, t)