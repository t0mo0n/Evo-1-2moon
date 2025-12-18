# evo1_server_json.py

import sys
import os
import asyncio
import logging
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1
from dataset.lerobot_dataset_pretrain_mp import NormalizationType



class Normalizer:
    def __init__(self, stats_or_path, normalization_type: NormalizationType = NormalizationType.BOUNDS):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)
        self.normalization_type = normalization_type
        print(f"Using normalization type: {self.normalization_type}")
        self.target_dim = 24

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        self.state_stats = self._prepare_stats(robot_stats.get("observation.state", {}), "observation.state")
        self.action_stats = self._prepare_stats(robot_stats.get("action", {}), "action")

    def _pad_vector(self, values, name):
        tensor = torch.tensor(values, dtype=torch.float32)
        length = tensor.shape[0]
        if length < self.target_dim:
            pad = torch.zeros(self.target_dim - length, dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=0)
        elif length > self.target_dim:
            raise ValueError(f"{name} length {length} exceeds expected {self.target_dim}")
        return tensor

    def _prepare_stats(self, stats_dict, stats_name):
        prepared = {}
        for key, values in stats_dict.items():
            prepared[key] = self._pad_vector(values, f"{stats_name}.{key}")
        return prepared

    def _stat_to_device(self, stats_dict, key, device, dtype):
        tensor = stats_dict.get(key)
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=dtype)

    def _normalize_tensor(self, tensor: torch.Tensor, stats_dict, clamp: bool) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError("Normal normalization selected but mean/std are missing in norm_stats.json")
            return (tensor - mean) / (std + eps)

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds normalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds normalization selected but min/max stats are missing in norm_stats.json")

        normalized = 2 * (tensor - low) / (high - low + eps) - 1
        if clamp:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def _denormalize_tensor(self, tensor: torch.Tensor, stats_dict) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError("Normal denormalization requested but mean/std stats are missing")
            return tensor * (std + eps) + mean

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds denormalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds denormalization requested but min/max stats are missing")

        return (tensor + 1.0) / 2.0 * (high - low + eps) + low

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._normalize_tensor(state, self.state_stats, clamp=True)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.view(1, -1)
        return self._denormalize_tensor(action, self.action_stats)


def load_model_and_normalizer(ckpt_dir):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = 32

    model = EVO1(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=True)
    model = model.to("cuda")

    normalization_type = config.get("normalization_type", NormalizationType.BOUNDS.value)
    normalizer = Normalizer(stats, normalization_type=normalization_type)
    return model, normalizer



def decode_image_from_list(img_list):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")



def infer_from_json_dict(data: dict, model, normalizer):
    device = "cuda"
    model_dtype = next(model.parameters()).dtype

  
    images = [decode_image_from_list(img) for img in data["image"]]
    assert len(images) == 3, "Must provide exactly 3 images."
    for img in images:
        assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"

 
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    
    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

    print(f"image_mask,{image_mask}")
    print(f"action_mask,{action_mask}")
    
    with torch.no_grad() and torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
        return action.cpu().numpy().tolist()


async def handle_request(websocket, model, normalizer):
    print("Client connected")
    try:
        async for message in websocket:
           
            json_data = json.loads(message)
            print(f"Received JSON observation")
            actions = infer_from_json_dict(json_data, model, normalizer)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")


    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
 

# === 启动服务 ===
if __name__ == "__main__":
    ckpt_dir = "Your/Path/To/Checkpoint"
    #Example: ckpt_dir = "/home/dell/checkpoints/Evo1/Evo1_MetaWorld/"

    port = 9000

    print("Loading EVO_1 model...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            "0.0.0.0", port, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())
