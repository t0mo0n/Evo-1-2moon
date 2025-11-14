# modeling_evo1.py

import os
import json
import torch
from collections import deque
from torch import Tensor
import numpy as np 
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.scripts.Evo1 import EVO1

class Normalizer:
    def __init__(self, stats_or_path):
        import json, torch
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            return x
        
        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        device = state.device
        dtype = state.dtype
        state_min = self.state_min.to(device=device, dtype=dtype)
        state_max = self.state_max.to(device=device, dtype=dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        device = action.device
        dtype = action.dtype
        action_min = self.action_min.to(device=device, dtype=dtype)
        action_max = self.action_max.to(device=device, dtype=dtype)
        if action.ndim == 1:
            action = action.view(-1, self.action_min.shape[0])
        return 0.5 * (action + 1.0) * (action_max - action_min + 1e-8) + action_min


class EVO1Policy(PreTrainedPolicy):
    config_class = Evo1Config
    name = "evo1"

    def __init__(self, config: Evo1Config, *, model_config_path: str, **kwargs):
        super().__init__(config)
       
        with open(model_config_path, "r") as f:
            model_cfg = json.load(f)
        model_cfg["device"] = config.device
        model_cfg["finetune_vlm"] = False
        model_cfg["finetune_action_head"] = False
        model_cfg["num_inference_timesteps"] = 32

        self.model = EVO1(model_cfg).eval().to(config.device)

        ckpt_dir = os.path.dirname(model_config_path)
        stats_path = os.path.join(ckpt_dir, "norm_stats.json")
        if not os.path.exists(stats_path):
            stats_path = os.path.join(ckpt_dir, "dataset_stats.json")
        self.normalizer = Normalizer(stats_path)
        print(f"Using Normalizer from: {stats_path}")

        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config: Evo1Config | None = None,
        strict: bool = False,
        **kwargs,
    ):
        ckpt_dir = str(pretrained_name_or_path)

        model_config_path = os.path.join(ckpt_dir, "model_config.json")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"NO model config: {model_config_path}")

        if config is None:
            config = cls.config_class(device="cuda")

        safe_kwargs = dict(kwargs)
        for k in ("config", "pretrained_name_or_path", "path", "strict"):
            safe_kwargs.pop(k, None)

        policy = cls(config, model_config_path=model_config_path, **safe_kwargs)

        ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"NO checkpoint file: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = raw["module"] if "module" in raw else raw
        missing, unexpected = policy.model.load_state_dict(state_dict, strict=strict)
        print(f"Weights loaded | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        return policy.eval().to(config.device)


    def reset(self):
        """Reset action queue when environment resets"""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return []

    def forward(self, batch: dict[str, Tensor]):
        return torch.tensor(0.0, device=self.config.device), {"loss": 0.0}
    

    @torch.no_grad()
    def predict_action_chunk(self, batch, noise=None):
        self.eval()

        def pad_state_to_24(x: torch.Tensor) -> torch.Tensor:
            missing = 24 - x.shape[-1]
            if missing > 0:
                pad = torch.zeros(*x.shape[:-1], missing, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
            return x

        state = batch[OBS_STATE]          
        state = pad_state_to_24(state)         
        norm_state = self.normalizer.normalize_state(state)

        prompt = batch["task"]
        
        RESIZE = 448
        images = []

        for k, v in batch.items():
            if k.startswith("observation.images."):
                img_chw = v.squeeze(0)  # (C,H,W)
                if isinstance(img_chw, np.ndarray):
                    img_chw = torch.from_numpy(img_chw)

                img_pil = to_pil_image(img_chw.clamp(0, 1))

                img_resized = img_pil.resize((RESIZE, RESIZE), resample=Image.BICUBIC)

                # Optional
                # img_resized.save(f"debug_model_input_{k}.jpg")

                img_tensor = ToTensor()(img_resized)

                images.append(img_tensor)

        num_img = len(images)
        mask_list = [1] * num_img + [0] * (3 - num_img)
        image_mask = torch.tensor(mask_list, dtype=torch.int32, device=self.config.device)

        if len(images) < 3:
            dummy = torch.zeros((3, RESIZE, RESIZE), dtype=torch.float32)
            images += [dummy] * (3 - len(images))
        images = images[:3]

        images = [img.to(self.config.device, dtype=torch.float32) for img in images]

        action_mask = torch.tensor([[1] * 6 + [0] * 18], dtype=torch.int32, device=self.config.device)

        with torch.no_grad() and torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = self.model.run_inference(
                images=images,
                image_mask=image_mask,
                prompt=prompt,
                state_input=norm_state,
                action_mask=action_mask,
            ) 
            
        actions = actions.reshape(1, -1, 24)
        actions = self.normalizer.denormalize_action(actions[0])

        return actions

    @torch.no_grad()
    def select_action(self, batch, noise=None):
        self.eval()

        if not hasattr(self, "_action_queue"):
            from collections import deque
            self._action_queue = deque(maxlen=self.config.n_action_steps)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise)   
            
            actions = actions[:35]
            # print("[DEBUG] INFERENCE ACTION:", actions)

            for i in range(actions.shape[0]):
                self._action_queue.append(actions[i])

        action = self._action_queue.popleft()

        action = action.reshape(-1) 
        # print("[DEBUG] Selected action:", [f"{x:.4f}" for x in action.tolist()])

        return action[:6]




