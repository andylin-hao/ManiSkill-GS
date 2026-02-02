import collections
import logging
from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
from tqdm import trange
import sys
import os
from mani_skill_gs.gs_renderer.gs_env import GSRenderWrapper
from mani_skill_gs.env_simulation.ms_env import *

cuda_bin = "/usr/local/cuda-12.4/bin"
current_path = os.environ.get("PATH", "")
os.environ["PATH"] = f"{cuda_bin}:{current_path}" if current_path else cuda_bin

sys.path.append("/usr/local/cuda-12.4/bin")

def setup_env():
    env = gym.make(
        "GSEnv-PutCubeOnPlate-v0", 
        robot_uids="my_franka", 
        render_mode="rgb_array", 
        reward_mode="none", 
        num_envs=1, 
        control_mode="pd_ee_target_delta_pose", 
        obs_mode="state",
        disable_env_checker=True,
        gs_kwargs={
            "render_interface": "gs_rlinf",
            "cache_table": True,
            "render_bs": 1,
            "device": "cuda",
        }
    )
    return env

def setup_policy(pretrained_path: str):
    """Setup OpenPI policy."""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if "pi05" in pretrained_path:
        config_name = "pi05_r2s2r"
    elif "pi0" in pretrained_path:
        config_name = "pi0_r2s2r"
    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name),
        pretrained_path,
        sample_kwargs={"num_steps": 5},
    )
    return policy

def main(
    pretrained_path: str,
    exp_name: str,
    num_episodes: int = 50,
    max_steps: int = 400,
):
    # Output base directory
    basedir = Path("outputs/eval/")
    
    # Logging
    log_dir = basedir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(str(log_file), mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Max Steps: %d", max_steps)
    logger.info("Creating env and loading policy...")
    env = setup_env()
    policy = setup_policy(pretrained_path)
    logger.info("Env and policy ready")

    total_successes = 0
    total_episodes = 0

    video_dir = basedir / f"video_{exp_name}"
    video_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(num_episodes):
        logger.info(f"Episode {ep + 1}/{num_episodes}")
        obs, info = env.reset()
        success = 0
        frames = []
        action_plan = collections.deque()

        for step in trange(max_steps):
            image = obs["sensor_data"]["3rd_view_camera"]["rgb"][0]
            state = obs["agent"]["qpos"][0]
            instruction = env.unwrapped.get_language_instruction()[0]
            batch = {
                "observation/image": image, # [480, 640, 3] torch.Tensor
                "observation/state": state, # [9] torch.Tensor [joint, end pose 0.04 (open) and -0.04 (close)]
                "prompt": instruction, # str
            }

            # Plan actions only if empty
            if not action_plan:
                action_chunk = policy.infer(batch)["actions"] # [5,7] np.array
                action_plan.extend(action_chunk)
            
            action = action_plan.popleft() # [7] np.array
            # TODO: for bingwen
            action
            obs, reward, terminated, truncated, info = env.step(action)

            frames.append(image)
            success = info.get("success")
            if success:
                break

        total_successes += success
        total_episodes += 1

        suffix = "success" if success else "failure"
        video_path = video_dir / f"episode_{ep:04d}_{suffix}.mp4"
        if len(frames) > 0:
            imageio.mimsave(str(video_path), np.stack(frames), fps=10)
        logger.info(f"Episode {ep}: success={success}, video={video_path}")

    env.close()

    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    logger.info("========== ManiSkill Eval ==========")
    logger.info(f"Total: {total_successes}/{total_episodes} episodes succeeded")
    logger.info(f"Success Rate: {success_rate:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name for log and video output prefix")
    parser.add_argument("--pretrained_path", type=str, required=False, help="OpenPI pretrained policy checkpoint path",
        default="checkpoints/pi05_sft_0124")
    parser.add_argument("--num_episodes", type=int, default=64, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")

    args = parser.parse_args()
    main(
        pretrained_path=args.pretrained_path,
        exp_name=args.exp_name,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )