"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import argparse
import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import matplotlib.pyplot as plt
import collections
import cv2

# === Server Interface ===
class OpenPIServer:
    def __init__(self, openpi_path: Union[str, Path]) -> Path:
        """
        A simple server for OpenPI models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        if "pi05" in openpi_path:
            config_name = "pi05_r2s2r"
        elif "pi0" in openpi_path:
            config_name = "pi0_r2s2r"
        self.policy = _policy_config.create_trained_policy(
            _config.get_config(config_name),
            openpi_path,
            sample_kwargs={"num_steps": 10},
        )
        self.action_plan = collections.deque()
        self.save_at_first = True


    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            agent_image = payload["images"][0]
            wrist_image = payload["images"][1]
            joints = payload["joints"]
            state = np.zeros(9)
            state = np.concatenate([
                joints, \
                np.array([payload["gripper_width"]][0]) / 2, \
                np.array([payload["gripper_width"]][0]) / 2
            ])
            # state = np.zeros(9)

            if self.save_at_first == True:
                cv2.imwrite(f"agent_image.png", agent_image[:,:,::-1])
                cv2.imwrite(f"wrist_image.png", wrist_image[:,:,::-1])
                self.save_at_first = False
            instruction = "pick up the cube and put it on the plate"
            batch = {
                "observation/image": agent_image,
                "observation/state": state,
                "prompt": instruction,
            }
            action_chunk = self.policy.infer(batch)["actions"]
            action_chunk[:,-1] = action_chunk[-1,-1]
            action = {"actions": action_chunk}
            return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

@dataclass
class DeployConfig:
    # fmt: off
    openpi_path: Union[str, Path] = "checkpoints/pi05_rl_100"               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 9623                                                    # Host Port

    # fmt: on

@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenPIServer(cfg.openpi_path)
    server.run(cfg.host, port=cfg.port)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy configuration")    
    parser.add_argument('--openpi_path', type=str, default="checkpoints/pi05_rl_100", help="HF Hub Path or local run directory")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Host IP Address")
    parser.add_argument('--port', type=int, default=9623, help="Host Port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = DeployConfig(
        openpi_path=args.openpi_path,
        host=args.host,
        port=args.port
    )
    deploy(config)