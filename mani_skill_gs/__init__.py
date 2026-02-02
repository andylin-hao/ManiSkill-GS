"""
GS (Gaussian Splatting) Environment Package

A package that provides ManiSkill environments with 3DGS rendering capabilities.
"""

from mani_skill_gs.gs_renderer.gs_env import GSRenderWrapper
from mani_skill_gs.env_simulation.ms_env import GSEnv

__all__ = ['GSRenderWrapper', 'GSEnv']


# register gs env
import gymnasium as gym
from mani_skill_gs.env_simulation import ms_env  
from mani_skill_gs.gs_renderer.gs_env import GSRenderWrapper

ENV_LIST = [
    "PutCubeOnPlate-v0"
]

def _make_gs_env_factory(env_id):
    def _factory(**kwargs):
        gs_kwargs = kwargs.pop("gs_kwargs")
        base_env = gym.make(env_id, **kwargs)
        env = GSRenderWrapper(base_env, **gs_kwargs)
        return env
    return _factory

for env_id in ENV_LIST:
    gs_env_id = f"GSEnv-{env_id}"
    gym.register(
        id=gs_env_id,
        entry_point=_make_gs_env_factory(env_id),
    )
    