import numpy as np
import sapien
from transforms3d.quaternions import mat2quat
from mani_skill_gs.env_simulation.ms_env import GSEnv
from motionplan.motionplanner import FR3MotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def move_to_pose(planner, pose, pose_name=""):
    """Move to pose with fallback to RRTConnect if screw planning fails."""
    res = planner.move_to_pose_with_screw(pose)
    if res == -1:
        if pose_name:
            print(f"Warning: Screw planning failed for {pose_name}, trying RRTConnect...")
        res = planner.move_to_pose_with_RRTConnect(pose)
        if res == -1:
            planner.close()
            return False
    return True


def get_obj_xyz(obj):
    """Extract position from the object."""
    xyz = obj.pose.p[0].cpu().numpy()
    return xyz


def get_grasp_pose(obj, env):
    # current TCP y-axis in world frame is used as target closing direction
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    # get grasp pose from OBB
    obb = get_actor_obb(obj)
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=np.array([0.0, 0.0, -1.0]),
        target_closing=target_closing,
        depth=0.025,
    )
    grasp_position = grasp_info["center"].copy()
    z_axis = grasp_info["approaching"]
    y_axis = grasp_info["closing"]
    y_proj = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_axis = y_proj / np.linalg.norm(y_proj)
    x_axis = np.cross(y_axis, z_axis)
    grasp_quat = mat2quat(np.stack([x_axis, y_axis, z_axis]).T)
    return sapien.Pose(p=grasp_position, q=grasp_quat)


def solve(env: GSEnv, data_collector, seed=None, debug: bool = False, vis: bool = False):
    """
    Pick-and-place with motion planning:
    1) grasp the object (cube) from above and
    2) place it on the plate.
    """
    # Setup planner
    obs, info = env.reset(seed=seed)
    for _ in range(10):
        dummy_action = np.zeros(7)
        dummy_action[6] = 1
        env.step(dummy_action)
    planner = FR3MotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        data_collector=data_collector,
    )

    # compute grasp object pose
    grasp_pose = get_grasp_pose(env.obj_cube, env)
    grasp_pose.p += np.array([0.0, 0.0, -0.02])
    # compute plate pose
    plate_xyz = get_obj_xyz(env.obj_plate)
    
    # Define waypoints as a list of dicts
    waypoints = [
        {
            "name": "Move to pre-grasp pose",
            "pose": sapien.Pose(p=grasp_pose.p + np.array([0, 0, 0.15]), q=grasp_pose.q),
            "action": None,
        },
        {
            "name": "Move to grasp pose",
            "pose": grasp_pose,
            "action": None,
        },
        {
            "name": "Close gripper",
            "pose": None,
            "action": {"type": "close_gripper", "t": 10},
        },
        {
            "name": "Move to post-grasp pose",
            "pose": sapien.Pose(p=grasp_pose.p + np.array([0, 0, 0.1]), q=grasp_pose.q),
            "action": None,
        },
        {
            "name": "Move to plate above pose",
            "pose": sapien.Pose(p=plate_xyz + np.array([0, 0, 0.1]), q=grasp_pose.q),
            "action": None,
        },
        {
            "name": "Move to plate release pose",
            "pose": sapien.Pose(p=plate_xyz + np.array([0, 0, 0.05]), q=grasp_pose.q),
            "action": None,
        },
        {
            "name": "Open gripper",
            "pose": None,
            "action": {"type": "open_gripper", "t": 15},
        },
        {
            "name": "Retract from plate",
            "pose": sapien.Pose(p=plate_xyz + np.array([0, 0, 0.15]), q=grasp_pose.q), 
            "action": None,
        }
    ]

    # Execute waypoints
    for waypoint in waypoints:
        # Move to pose if pose is provided
        if waypoint["pose"] is not None:
            res = planner.move_to_pose_with_screw(waypoint["pose"])
            if res == -1:
                planner.close()
                return False
        
        # Execute action if action is provided
        if waypoint["action"] is not None:
            if waypoint["action"]["type"] == "close_gripper":
                planner.close_gripper(t=waypoint["action"]["t"])
            elif waypoint["action"]["type"] == "open_gripper":
                planner.open_gripper(t=waypoint["action"]["t"])
            else:
                raise RuntimeError(f"Invalid action name: {waypoint['name']}")

    planner.close()
    
    # Check success
    return env.is_success()