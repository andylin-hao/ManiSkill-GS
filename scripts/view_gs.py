import mani_skill_gs 
import gymnasium as gym

if __name__ == "__main__":
    # env build
    env = gym.make(
        "GSEnv-PutCubeOnPlate-v0", 
        robot_uids="my_franka", 
        render_mode="rgb_array", 
        reward_mode="none", 
        num_envs=16, 
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
    # reset
    gs_imgs = []
    obs, info = env.reset()
    # test viewer
    print("Opening viewer...")
    env.open_viewer()
    while True:
        action = env.action_space.sample()
        env.step(action)

