import mani_skill_gs
import gymnasium as gym
from tqdm import trange
from mani_skill_gs.utils.utils_img import concat_images
import torch
import os
import imageio
import numpy as np

if __name__ == "__main__":
    # env
    env = gym.make(
        "GSEnv-PutCubeOnPlate-v0", 
        robot_uids="my_franka", 
        render_mode="rgb_array", 
        reward_mode="none", 
        num_envs=4, 
        control_mode="pd_ee_target_delta_pose", 
        obs_mode="state",
        disable_env_checker=True,
        gs_kwargs={
            "render_interface": "gs_rlinf",
            "cache_table": True,
            "render_bs": 4,
            "device": "cuda",
        }
    )

    # reset
    gs_imgs = []
    obs, info = env.reset()

    # save videos
    for step_idx in trange(30, desc="Running steps"):
        # render
        render_result = obs["sensor_data"]["3rd_view_camera"]
        img_gs = render_result["rgb"].numpy()  # [num_envs, H, W, 3]
        merged_gs = concat_images(img_gs)
        gs_imgs.append(merged_gs)
        # interact
        action = torch.randn((env.num_envs, 7), device=env.device) * 0.02
        obs, reward, terminated, truncated, info = env.step(action)
        
    # video
    os.makedirs("outputs/videos", exist_ok=True)
    gs_frames = np.stack(gs_imgs, axis=0)
    output_path = "outputs/videos/test_rlinf_interface.mp4" 
    imageio.mimsave(output_path, gs_frames, fps=10)
    print(f"Saved GS video to {output_path} with {len(gs_imgs)} frames")
