import mani_skill_gs
import gymnasium as gym
import argparse
from motionplan.solutions.pick_cube import solve
from motionplan.data_collector import DataCollector
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Collect motion-planning trajectories and export to LeRobot format.")
    parser.add_argument("--num-traj", type=int, default=25, help="Number of successful trajectories to collect")
    args = parser.parse_args()

    # Build ManiSkill env and wrap with GS renderer
    env = gym.make(
        "GSEnv-PutCubeOnPlate-v0", 
        robot_uids="my_franka", 
        render_mode="rgb_array", 
        reward_mode="none", 
        num_envs=1, 
        control_mode="pd_ee_target_delta_pose",  # pd_ee_delta_pose, pd_ee_target_delta_pose
        obs_mode="state",
        disable_env_checker=True,
        gs_kwargs={
            "render_interface": "gs_rlinf",
            "cache_table": True,
            "render_bs": 1,
            "device": "cuda",
        }
    )

    # LeRobot dataset
    lerobot_dataset = None
    lerobot_repo_id = "RLinf/GSEnv-PutCubeOnPlate-v0"
    lerobot_root =  f"datasets/{lerobot_repo_id}"
    
    # Data collector (images, state, action, instruction)
    data_collector = DataCollector(env)

    # Collect trajectories
    successes = []
    passed = 0
    idx = 0
    
    # Collect trajectories
    print(f"Starting data collection: {args.num_traj} trajectories")
    while passed < args.num_traj:
        success = bool(solve(env, data_collector, seed=idx, debug=False, vis=False))
        if success:
            successes.append(True)
            # Episode data from DataCollector
            episode_data = data_collector.save_data()
            # Create LeRobot dataset on first successful episode
            if lerobot_dataset is None:
                first_img = episode_data["image"][0]
                width, height = first_img.size
                # create lerobot dataset
                lerobot_dataset = LeRobotDataset.create(
                    repo_id=lerobot_repo_id,
                    root=lerobot_root,
                    robot_type="franka",
                    fps=10,
                    features={
                        "observation.image": {
                            "dtype": "video",
                            "shape": (height, width, 3),
                            "names": ["height", "width", "channel"],
                        },
                        "observation.state": {
                            "dtype": "float32",
                            "shape": (9,),
                        },
                        "actions": {
                            "dtype": "float32",
                            "shape": (7,),
                        },
                    },
                    use_videos=True,
                    image_writer_threads=4,
                    image_writer_processes=2,
                )

            # Write current episode to LeRobot dataset
            images = episode_data["image"]
            actions = episode_data["action"]  
            states = episode_data["state"]   
            instruction = episode_data["instruction"]
            task_text = str(instruction)

            # DataCollector only updates on step(), so image/state/action lengths must match
            T = min(len(images), len(actions), len(states))
            if not (len(images) == len(states) == len(actions)):
                print(
                    f"[LeRobot] Length mismatch: images={len(images)}, "
                    f"states={len(states)}, actions={len(actions)}. Using T={T}."
                )
            for t in range(T):
                # [H,W,3], uint8
                img_np = np.array(images[t])  
                # State is qpos
                state = states[t].astype(np.float32)
                action = actions[t].astype(np.float32)
                lerobot_dataset.add_frame(
                    {
                        "observation.image": img_np,
                        "observation.state": state,
                        "actions": action,
                        "task": task_text,
                    }
                )
            # save
            lerobot_dataset.save_episode()
            passed += 1
            print(f"Collected trajectory {passed}/{args.num_traj} (idx={idx})")
            
            if passed >= args.num_traj:
                break
        else:
            successes.append(False)
            data_collector.clear_data()
        idx += 1
    
    print("\nData collection complete!")
    print(f"  Total trajectories collected: {passed}")
    print(f"  Success rate: {np.mean(successes) if successes else 0:.2%}")
    env.close()
