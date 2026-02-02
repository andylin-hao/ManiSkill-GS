import torch
import torch.nn as nn
from pathlib import Path
import threading
import time
from typing import Optional

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.utils import writer

from mani_skill_gs.gs_renderer.gs_model import GaussModel


class GaussViewer:
    """
    Interactive viewer for Gaussian Splatting models.
    
    This class provides functionality to visualize and interact with
    Gaussian Splatting models using the Nerfstudio viewer.
    
    Example:
        >>> from gs.gs_params import GaussParams
        >>> device = torch.device("cuda:0")
        >>> model = GaussModel(device)
        >>> params = GaussParams.from_ply("scene.ply", device)
        >>> model.load_params(params)
        >>> viewer = GaussViewer(model)
        >>> viewer.open()
    """
    
    def __init__(
        self,
        gs_model: GaussModel,
        websocket_port: int = 7007,
        websocket_host: str = "0.0.0.0",
        jpeg_quality: int = 95,
        max_res: int = 512,
        datapath: Optional[Path] = None,
    ):
        """
        Initialize the viewer.
        
        Args:
            gs_model: GaussModel instance with loaded parameters
            websocket_port: Port for the viewer websocket
            websocket_host: Host for the viewer websocket
            jpeg_quality: JPEG quality for rendered images (1-100)
            max_res: Maximum resolution for rendering
            datapath: Path for viewer data (defaults to "gs/data")
        """
        self.gs_model = gs_model
        self.gauss_params = gs_model.get_params()
        self.websocket_port = websocket_port
        self.websocket_host = websocket_host
        self.jpeg_quality = jpeg_quality
        self.max_res = max_res
        self.datapath = datapath if datapath else Path("gs/data")
        
    def _create_minimal_dataset(self) -> InputDataset:
        """
        Create a minimal dataset from GS params for viewer.
        
        Constructs a dataset structure required by Nerfstudio viewer from
        the Gaussian parameters. Computes scene bounding box and sets up
        a default camera configuration.
        
        Returns:
            InputDataset object configured for the viewer
            
        Note:
            The scene box is computed with 10% padding around the Gaussian bounds.
            A default camera is positioned above the scene center.
        """
        # Work on a detached CPU tensor to avoid autograd graph and deepcopy issues
        means = self.gauss_params.means.detach().cpu()
        min_bounds = means.min(dim=0)[0]
        max_bounds = means.max(dim=0)[0]
        center = (min_bounds + max_bounds) / 2
        size = (max_bounds - min_bounds).max()
        padding = size * 0.1
        aabb = torch.stack([
            center - size / 2 - padding,
            center + size / 2 + padding
        ])
        # Ensure aabb is a leaf tensor (no grad history) so deepcopy works
        aabb = aabb.detach().clone()
        scene_box = SceneBox(aabb=aabb)
        
        c2w = torch.eye(4)[:3, :]
        c2w[2, 3] = (center[2] + size).item()
        
        cameras = Cameras(
            camera_to_worlds=c2w.unsqueeze(0),
            fx=torch.tensor([500.0]),
            fy=torch.tensor([500.0]),
            cx=torch.tensor([320.0]),
            cy=torch.tensor([240.0]),
            width=640,
            height=480,
            camera_type=CameraType.PERSPECTIVE,
        )
        
        dataparser_outputs = DataparserOutputs(
            image_filenames=[],
            cameras=cameras,
            alpha_color=None,
            scene_box=scene_box,
            metadata={},
        )
        
        dataset = InputDataset(dataparser_outputs, scale_factor=1.0)
        return dataset
    
    def _setup_model_for_viewer(self):
        """
        Setup model for viewer (configure step for SH degree).
        """
        model = self.gs_model.model
        if not hasattr(model, 'step'):
            has_rest_features = torch.any(self.gauss_params.features_rest != 0)
            if has_rest_features:
                sh_degree_interval = model.config.sh_degree_interval
                sh_degree = model.config.sh_degree
                model.step = sh_degree_interval * sh_degree
                print(f"Detected SH features, using step={model.step} for full sh_degree={sh_degree}")
            else:
                model.step = 0
                print("Only DC features detected, using step=0")
    
    def _initialize_writer(self):
        """
        Initialize writer (required for rendering speed estimation).
        """
        if not writer.is_initialized():
            from nerfstudio.configs.base_config import LoggingConfig, LocalWriterConfig
            logging_config = LoggingConfig(
                local_writer=LocalWriterConfig(enable=False),
                max_buffer_size=20,
                steps_per_log=10,
            )
            writer.setup_local_writer(
                config=logging_config,
                max_iter=1,
                banner_messages=None,
            )

    def _wrap_model_for_viewer(self):
        """
        Wrap the model's get_outputs_for_camera so that it returns
        unbatched outputs (H, W, C) instead of (B, H, W, C) for B=1.

        Nerfstudio's viewer expects images without a leading batch
        dimension in some places, which can cause shape mismatches
        when padding. This wrapper keeps training behavior intact,
        but for the viewer we squeeze a singleton batch dimension.
        """
        model = self.gs_model.model

        # Avoid wrapping multiple times
        if getattr(model, "_gsviewer_wrapped", False):
            return

        if not hasattr(model, "get_outputs_for_camera"):
            return

        orig_get_outputs = model.get_outputs_for_camera

        def wrapped_get_outputs(cameras, *args, **kwargs):
            outputs = orig_get_outputs(cameras, *args, **kwargs)
            if not isinstance(outputs, dict):
                return outputs

            new_outputs = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1:
                    new_outputs[k] = v[0]
                else:
                    new_outputs[k] = v
            # Nerfstudio viewer expects outputs["depth"] with shape (H, W, 1)
            if new_outputs.get("depth") is None and "rgb" in new_outputs:
                rgb = new_outputs["rgb"]
                h, w = rgb.shape[0], rgb.shape[1]
                new_outputs["depth"] = torch.zeros(
                    (h, w, 1), dtype=rgb.dtype, device=rgb.device
                )
            return new_outputs

        model.get_outputs_for_camera = wrapped_get_outputs
        model._gsviewer_wrapped = True
    
    def open(self):
        """
        Open the interactive viewer.
        
        This method will start the viewer server and block until interrupted.
        Press Ctrl+C to stop the viewer.
        """
        self.gs_model.model = self.gs_model.model.to(self.gs_model.device)
        for k, v in self.gs_model.model.gauss_params.items():
            if isinstance(v, torch.Tensor):
                self.gs_model.model.gauss_params[k] = nn.Parameter(v.to(self.gs_model.device))

        # Ensure model outputs have the expected shape for the viewer
        self._wrap_model_for_viewer()

        self._setup_model_for_viewer()
        self._initialize_writer()
        
        dataset = self._create_minimal_dataset()
        
        class SimpleDataManager:
            def __init__(self, dataset, datapath):
                self.train_dataset = dataset
                self.eval_dataset = dataset
                self.includes_time = False
                self.datapath = datapath
            
            def get_datapath(self):
                return self.datapath
        
        datamanager = SimpleDataManager(dataset, self.datapath)
        
        class SimplePipeline:
            def __init__(self, model, datamanager):
                self.model = model
                self.datamanager = datamanager
        
        pipeline = SimplePipeline(self.gs_model.model, datamanager)
        train_lock = threading.Lock()
        
        viewer_config = ViewerConfig(
            websocket_port=self.websocket_port,
            websocket_host=self.websocket_host,
            jpeg_quality=self.jpeg_quality,
        )
        viewer = Viewer(
            config=viewer_config,
            log_filename=Path("gs/viewer_log.txt"),
            datapath=self.datapath,
            pipeline=pipeline,
            trainer=None,
            train_lock=train_lock,
            share=False,
        )
        
        viewer.control_panel._max_res.value = self.max_res
        
        viewer.init_scene(
            train_dataset=dataset,
            train_state="completed",
            eval_dataset=dataset,
        )
        viewer.update_scene(step=0)
        
        print(f"Viewer started! {viewer.viewer_info[0]}")
        print("Press Ctrl+C to stop the viewer")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nViewer stopped.")

