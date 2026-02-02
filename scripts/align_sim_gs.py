import mani_skill_gs
import gymnasium as gym
import imageio
import numpy as np
import os
from mani_skill_gs.utils.utils_img import concat_images, blend_images_transparent
from mani_skill_gs.gs_renderer.gs_env import GSRenderWrapper
from tqdm import trange
import torch


if __name__ == "__main__":
    # env build
    env = gym.make(
        "PutCubeOnPlate-v0",
        robot_uids="my_franka",
        render_mode="rgb_array",
        reward_mode="none",
        num_envs=4,
        control_mode="pd_ee_target_delta_pose",
        obs_mode="state",
        disable_env_checker=True,
    )

    # gs wrapper
    env : GSRenderWrapper = GSRenderWrapper(
        env, 
        render_interface="gs+env",
        cache_table=True,
        render_bs=8, # 4,8,16
        device="cuda", 
    )

    # reset
    gs_imgs = []
    env_imgs = []
    blended_imgs = []
    obs, info = env.reset()

    # env run
    for _ in trange(20):
        # render
        render_result = obs["sensor_data"]["3rd_view_camera"]
        img_gs = render_result["gs"].numpy()  # [num_envs, H, W, 3]
        img_env = render_result["env"].numpy()  # [num_envs, H, W, 3]
        merged_gs = concat_images(img_gs)
        merged_env = concat_images(img_env)
        blended = blend_images_transparent(merged_env, merged_gs, alpha=0.5)
        gs_imgs.append(merged_gs)
        env_imgs.append(merged_env)
        blended_imgs.append(blended)
        # interact
        action = torch.randn((env.num_envs, 7), device=env.device) * 0.02
        obs, reward, terminated, truncated, info = env.step(action)

    # save videos
    gs_frames = np.stack(gs_imgs, axis=0)
    env_frames = np.stack(env_imgs, axis=0)
    blended_frames = np.stack(blended_imgs, axis=0)

    output_dir = "outputs/videos"
    os.makedirs(output_dir, exist_ok=True)

    # save gs video
    gs_path = os.path.join(output_dir, "align_sim_gs_gs.mp4")
    imageio.mimsave(gs_path, gs_frames, fps=10)

    # save sim(env) video
    env_path = os.path.join(output_dir, "align_sim_gs_env.mp4")
    imageio.mimsave(env_path, env_frames, fps=10)

    # save blended video
    blended_path = os.path.join(output_dir, "align_sim_gs_blend.mp4")
    imageio.mimsave(blended_path, blended_frames, fps=10)

    print(
        f"Saved gs video to {gs_path}, "
        f"sim video to {env_path}, "
        f"and blended video to {blended_path} "
        f"with {len(blended_imgs)} frames"
    )

