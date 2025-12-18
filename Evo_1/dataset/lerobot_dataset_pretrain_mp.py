#use lerobot_dataset_pretrain_mp.py for multithreading load dataset
import os
import io
import torch
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm  
from typing import List, Union, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToTensor
from collections.abc import Iterable
import multiprocessing as mp
import logging
import pickle
from enum import Enum

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99" 

def compute_lerobot_normalization_stats_from_minmax(jsonl_path, relative_path=None):
    state_mins, state_maxs = [], []
    action_mins, action_maxs = [], []

    if relative_path is None:
        with open(jsonl_path, "r") as f:
            for line in tqdm(f, desc="Extracting min/max"):
                obj = json.loads(line)
                stats = obj.get("stats", {})
                try:
                    state_mins.append(stats["observation.state"]["min"])
                    state_maxs.append(stats["observation.state"]["max"])
                    action_mins.append(stats["action"]["min"])
                    action_maxs.append(stats["action"]["max"])
                except Exception as e:
                    print(f"skipping abnormal line: {e}")
    else:
        with open(jsonl_path, "r") as f:
            for line in tqdm(f, desc="Extracting min/max"):
                obj = json.loads(line)
                stats = obj.get("stats", {})
                try:
                    state_mins.append(stats["observation.state"]["min"])
                    state_maxs.append(stats["observation.state"]["max"])
                except Exception as e:
                    print(f"skipping abnormal line: {e}")

        with open(relative_path, "r") as f:
            for line in tqdm(f, desc="Extracting relative action min/max"):
                obj = json.loads(line)
                stats = obj.get("stats", {})
                try:
                    action_mins.append(stats["action_relative"]["min"])
                    action_maxs.append(stats["action_relative"]["max"])
                except Exception as e:
                    print(f"skipping abnormal line: {e}")


    state_min_global = np.min(np.array(state_mins), axis=0).tolist()
    state_max_global = np.max(np.array(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.array(action_mins), axis=0).tolist()
    action_max_global = np.max(np.array(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }

def merge_lerobot_stats(stats_list: List[Dict[str, Dict[str, List[float]]]]) -> Dict:
    if not stats_list:
        raise ValueError("stats_list cannot be empty when merging normalization stats")

    merged = {}
    for space in ["observation.state", "action"]:
        merged[space] = {}
        metric_keys = set()
        for stats in stats_list:
            metric_keys.update(stats.get(space, {}).keys())

        for metric in metric_keys:
            tensors = [np.array(stats[space][metric]) for stats in stats_list if metric in stats.get(space, {})]
            if not tensors:
                continue
            stacked = np.stack(tensors)

            if metric in {"min", "q01"}:
                merged_val = np.min(stacked, axis=0)
            elif metric in {"max", "q99"}:
                merged_val = np.max(stacked, axis=0)
            elif metric in {"mean", "std"}:
                merged_val = np.mean(stacked, axis=0)
            else:
                merged_val = stacked[0]

            merged[space][metric] = merged_val.tolist()

    return merged


def _process_parquet_file_worker(args):
    parquet_path, arm_name, dataset_name, dataset_config, dataset_path, task_mapping, action_horizon, max_samples_per_file, cache_dir = args
    
    use_delta_action = dataset_config.get('use_delta_action', True)
    
    try:
        view_map = dataset_config.get('view_map', None)
        if not view_map:
            logging.info(f"did not find view_map for '{arm_name}-{dataset_name}', use default mapping")
            default_keys = ["image_1", "image_2", "image_3"]
            view_map = {key: f"observation.images.{key}" for key in default_keys}

        df = pd.read_parquet(parquet_path)

        last_row = df.iloc[-1:]  
        padding_rows = pd.concat([last_row] * action_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)

        if max_samples_per_file is not None:
            df = df.head(max_samples_per_file)

        episode_files = []
        for i in range(len(df) - action_horizon + 1): 
            start_idx = i
            end_idx = i + action_horizon
            
      
            cache_subdir = cache_dir / arm_name / dataset_name / parquet_path.parent.name / parquet_path.stem
            cache_filename = f"{start_idx}_{end_idx}.pkl"
            cache_filepath = cache_subdir / cache_filename
            
            
            if cache_filepath.exists():
                episode_files.append(str(cache_filepath))
                continue
            
            logging.info(f"build {cache_filename}")
            sub_df = df.iloc[i: i + action_horizon]
            
            # convert actions to relative actions from first state
            init_state = sub_df.iloc[0].get("observation.state", None)
            if use_delta_action:
                actions = np.stack(sub_df["action"].to_list())
                if init_state is not None:
                    min_dim = min(len(init_state), actions.shape[1])
                    init_state_broadcast = np.zeros_like(actions)
                    init_state_broadcast[:, :min_dim] = init_state[:min_dim]
                    relative_actions = actions - init_state_broadcast
                else:
                    logging.warning(f"File {parquet_path} at index {i} has no 'observation.state', actions will not be converted to relative.")
                    relative_actions = actions
                actions = relative_actions.tolist()
            else:
                actions = sub_df["action"].to_list()
            
            video_paths = {}
            base_video_path = dataset_path / "videos" / parquet_path.parent.name

            for view_key, view_folder in view_map.items():
                full_path = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
                logging.info(f"full_path {full_path}")
                if full_path.exists():
                    video_paths[view_key] = str(full_path)
                else:
                    logging.warning(f"missing video file: {full_path}")
            
            
            task_index = sub_df.iloc[0].get("task_index", None)
            if task_index is not None and task_index in task_mapping:
                prompt = task_mapping[task_index]
            else:
                logging.info(f"cannot find task description from task_index={task_index}")
                prompt = ""

            episode = {
                "arm_key": arm_name,
                "dataset_key": dataset_name,
                "prompt": prompt,
                "state": init_state,
                "action": actions,
                "video_paths": video_paths,
                "timestamp": sub_df.iloc[0].get("timestamp", None),
            }
            
            cache_subdir.mkdir(parents=True, exist_ok=True)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(episode, f)
            
            episode_files.append(str(cache_filepath))
        return episode_files, None 
        
    except Exception as e:
        error_msg = f"Error processing file {parquet_path}: {str(e)}"
        logging.error(error_msg)
        return [], error_msg

class LeRobotDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        video_backend: str = "av", # TODO: 
        action_horizon: int = 50,
        video_backend_kwargs: Dict[str, Any] = None,
        binarize_gripper: bool = False,
        cache_dir: Union[str, Path] = None,  
        use_augmentation: bool = False
    ):
        self.config = config

        sorted_datasets = sorted(self.config['data_groups'].keys())
        self.arm_to_embodiment_id = {key: i for i, key in enumerate(sorted_datasets)}

        self.max_action_dim = config['max_action_dim']
        self.max_state_dim = config['max_state_dim']
        self.max_views = config['max_views']

        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation


        if cache_dir is None:
            self.cache_dir = Path("/home/tomoon/codes/Evo-VLA/train_data_cache")
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []  
        self.arm2stats_dict = {}
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}  

        if self.video_backend == "decord" and not self.video_backend_kwargs:
            self.video_backend_kwargs = {"ctx": "cpu"}  

        self._load_metadata()
        self._load_trajectories()

        self.basic_transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.aug_transform = T.Compose([
            T.RandomResizedCrop(448, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC), 
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor()
        ])

    def _load_metadata(self):
     
        self.episodes = []
        self.tasks = {}
        self.normalization_type = self.config.get("normalization_type", NormalizationType.BOUNDS.value)
        logging.info(f"Using normalization type: {self.normalization_type}")
        norm_stats_list = []

        # for arms
        for arm_name, arm_config in self.config['data_groups'].items():
            print(f"  -- Processing arm group: '{arm_name}'")

            norm_arm_list = []
            self.tasks[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                print(f"    -- Processing dataset: '{dataset_name}'")
                print(f"    -- Dataset config: {dataset_config}")
                dataset_tasks = []
                path_str = dataset_config['path']
                dataset_path = Path(path_str)
                tasks_path = dataset_path / "meta" / "tasks.jsonl"
                if tasks_path.exists():
                    dataset_tasks = pd.read_json(tasks_path, lines=True).to_dict("records")
                    task_index_to_task = {
                        task_obj["task_index"]: task_obj["task"]
                        for task_obj in dataset_tasks
                        if "task_index" in task_obj and "task" in task_obj
                    }
                    self.tasks[arm_name][dataset_name] = task_index_to_task
                else:
                    raise FileNotFoundError(f"tasks file not found: {tasks_path}")
                
                episodes_path = dataset_path / "meta" / "episodes.jsonl"
                if episodes_path.exists():
                    self.episodes += pd.read_json(episodes_path, lines=True).to_dict("records")

     
                stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
                stats_path_after_compute = dataset_path / "meta" / "stats.json"
                use_delta_action = dataset_config.get('use_delta_action', True)
                if use_delta_action:
                    stat_relative_path = dataset_path / "meta" / "episodes_stats_action_relative.jsonl"
                else:
                    stat_relative_path = None
                if stats_path_after_compute.exists():
                    print(f"already have stats file: {stats_path_after_compute}")
                    with open(stats_path_after_compute, "r") as f:
                        stats = json.load(f)
                    norm_arm_list.append(stats)
                elif stats_path.exists():
                    stats = compute_lerobot_normalization_stats_from_minmax(stats_path, stat_relative_path)
                   
                    with open(stats_path_after_compute, "w") as f:
                        json.dump(stats, f, indent=4)
               
                    print(f"computed stats and saved to: {stats_path_after_compute}")
                    norm_arm_list.append(stats)
                else:
                    raise FileNotFoundError(f"normalization stats file not found: {stats_path}")
            

            self.arm2stats_dict[arm_name] = merge_lerobot_stats(norm_arm_list)


    def _load_trajectories(self):

        

        parquet_process_units = []
        for arm_name, arm_config in self.config['data_groups'].items():
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = dataset_config.get('path', None)
                if dataset_path is None:
                    raise ValueError(f"Dataset path for '{arm_name}-{dataset_name}' is not configured, please check the config")
                dataset_path = Path(dataset_path)
                parquet_files = list(dataset_path.glob("data/*/*.parquet"))
                
                task_mapping = self.tasks[arm_name][dataset_name]
                
                for parquet_path in parquet_files:
                    parquet_process_units.append((
                        parquet_path, 
                        arm_name, 
                        dataset_name, 
                        dataset_config, 
                        dataset_path,
                        task_mapping,  
                        self.action_horizon,
                        self.max_samples_per_file,
                        self.cache_dir  
                    ))

       
        print(f"total {len(parquet_process_units)} parquet files to process")
        
   
        num_processes = min(16, len(parquet_process_units))

        print(f"Using {num_processes} processes for concurrent processing")
        
 
        with mp.Pool(processes=num_processes) as pool:
            
            total_episodes = 0
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files to cache") as pbar:
                for episode_files, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                    else:
                        self.data.extend(episode_files)  
                        total_episodes += len(episode_files)
                    
                    pbar.set_postfix({
                        'episodes_this_file': len(episode_files),
                        'total_episodes': total_episodes
                    })
                    pbar.update(1)
        
        print(f"Data processing completed, total {len(self.data)} files generated")


    def _pad_tensor(
        self, 
        source_tensor: torch.Tensor, 
        max_dim: int
    ) -> (torch.Tensor, torch.Tensor):

        source_dim = source_tensor.shape[-1]
        
        if source_tensor.dim() > 1:
            padded_shape = (*source_tensor.shape[:-1], max_dim)
        else:
            padded_shape = (max_dim,)

        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)

        data_slice = (..., slice(0, source_dim))
        
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
            
        return padded_tensor, mask


    def _load_video_frame(self, video_paths: dict, timestamp: float) -> List[Image.Image]:
    
        frames = []
        for view, path in video_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"video file not found: {path}")
            
            if self.video_backend == "decord":
                import decord

                try:
                    ctx = self.video_backend_kwargs.get("ctx", "cpu")
                    if ctx == "cpu":
                        ctx = decord.cpu(0)
                    elif ctx == "gpu":
                        ctx = decord.gpu(0)
                    logging.info(f"Using video backend {self.video_backend}, context: {ctx}")
                    vr = decord.VideoReader(path, ctx=ctx)
                    logging.info(f"Successfully opened video file: {path}")
                    fps = vr.get_avg_fps()
                    logging.info(f"Video {path} FPS: {fps}")
                    if fps is None or np.isnan(fps):
                        raise ValueError(f"Unable to read FPS, video may be corrupted: {path}")

                    frame_idx = int(timestamp * fps)
                    logging.info(f"Reading video {path} frame index: {frame_idx} (timestamp: {timestamp}, fps: {fps})")
                    if frame_idx >= len(vr):
                        logging.info(f"the requested frame index exceeds video length: frame_idx={frame_idx}, len={len(vr)}. Using last frame instead.")
                        
                        frame_idx = len(vr) - 1

                    frame = vr[frame_idx].asnumpy()
                    frames.append(Image.fromarray(frame))
                    logging.info(f"Successfully read video frame: {path}, frame index: {frame_idx}")

                except Exception as e:
                    logging.info(f"Failed to read video file: {path}")
                    logging.info(f"Error message: {str(e)}")
                    raise

            elif self.video_backend == "av":
                import av
                try:
                    with av.open(path) as container:
                        for frame in container.decode(video=0):
                            if frame.time >= timestamp:
                                frames.append(Image.fromarray(frame.to_ndarray(format='rgb24')))
                                break

                except Exception as e:
                    print(f"Failed to read video file: {path}")
                    print(f"Error message: {str(e)}")
                    raise
            else:
                raise NotImplementedError(f"Video backend {self.video_backend} not implemented")
        
        return frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cache_filepath = self.data[idx]
        
        try:
            with open(cache_filepath, 'rb') as f:
                item = pickle.load(f)
        except Exception as e:
            logging.info(f"cannot load cache file {cache_filepath}: {str(e)}")
            
            return self[random.randint(0, len(self.data)-1)]
 
        
        arm_key = item["arm_key"]
        dataset_key = item["dataset_key"]
        embodiment_id = self.arm_to_embodiment_id[arm_key]

 
        try:
            frames = self._load_video_frame(item["video_paths"], item["timestamp"])
        except Exception as e:
      
            logging.info(f"skipping sample that cannot decode video {self.data[idx]}: {e}")
            return self[random.randint(0, len(self.data)-1)]  

        images = frames


        if self.use_augmentation:
           
            images = [
                self.aug_transform(img) if random.random() < 0.5 else self.basic_transform(img)
                for img in images
            ]
        else:
         
            images = [self.basic_transform(img) for img in images]

 
        num_real_views = len(images)
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True 


        while len(images) < self.max_views:
           
            if len(images) == 0:
                dummy_image = torch.zeros(3, 448, 448)
                logging.info("Warning: Image list is empty, using zero tensor for padding")
            else:
                dummy_image = torch.zeros_like(images[0]) 
            images.append(dummy_image)

        images = torch.stack(images)


        if item["state"] is None:
            raise ValueError("missing observation.state, please check data integrity")
        
    

        try:
            norm_stats = self.arm2stats_dict[arm_key]
        except KeyError:
        
            raise KeyError(f"Normalization stats not found for arm_key={arm_key} and dataset_key={dataset_key}")

        

        state = torch.tensor(item["state"], dtype=torch.float32)
        state = self._normalize_tensor(
            tensor=state,
            stats_dict=norm_stats["observation.state"],
            stats_name="observation.state",
            normalization_type=self.normalization_type,
            apply_zero_mask=False,
        )

        state_padded, state_mask = self._pad_tensor(
            state, self.max_state_dim
        )


        if item["action"] is None:
            raise ValueError("missing action, please check data integrity")

  
        action = torch.from_numpy(np.stack(item["action"])).float()
        action = self._normalize_tensor(
            tensor=action,
            stats_dict=norm_stats["action"],
            stats_name="action",
            normalization_type=self.normalization_type,
            apply_zero_mask=True,
        )

        action_padded, action_mask = self._pad_tensor(
            action, self.max_action_dim
        )

        prompt = item["prompt"] if item["prompt"] is not None else ""
        
        return {
            "images": images,
            "image_mask": image_mask,
            "prompt": prompt,
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(embodiment_id, dtype=torch.long)
        }

    def _normalize_tensor(self, tensor, stats_dict, stats_name, normalization_type, apply_zero_mask=False):
        """Normalize tensor according to stats and requested normalization type."""
        eps = 1e-8
        device = tensor.device

        def _stat_tensor(key):
            values = stats_dict.get(key)
            if values is None:
                return None
            return torch.tensor(values, dtype=torch.float32, device=device)

        if normalization_type == NormalizationType.NORMAL:
            mean = _stat_tensor("mean")
            std = _stat_tensor("std")
            if mean is None or std is None:
                raise ValueError(
                    f"Missing mean/std stats for {stats_name}; please regenerate stats.json with these metrics."
                )
            return (tensor - mean) / (std + eps)

        if normalization_type == NormalizationType.BOUNDS:
            low = _stat_tensor("min")
            high = _stat_tensor("max")
            if low is None or high is None:
                raise ValueError(f"Missing min/max stats for {stats_name}; cannot apply bounds normalization.")
        elif normalization_type == NormalizationType.BOUNDS_Q99:
            low = _stat_tensor("q01")
            high = _stat_tensor("q99")
            if low is None or high is None:
                logging.warning(
                    "Missing q01/q99 for %s, falling back to min/max bounds normalization.",
                    stats_name,
                )
                low = _stat_tensor("min")
                high = _stat_tensor("max")
            if low is None or high is None:
                raise ValueError(
                    f"Missing q01/q99 (or min/max fallback) for {stats_name}; cannot apply bounds_q99 normalization."
                )
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

        diff = high - low
        normalized = 2 * (tensor - low) / (diff + eps) - 1
        normalized = torch.clamp(normalized, -1.0, 1.0)

        if apply_zero_mask:
            zeros_mask = diff.abs() < eps
            while zeros_mask.dim() < normalized.dim():
                zeros_mask = zeros_mask.unsqueeze(0)
            normalized = torch.where(zeros_mask, torch.zeros_like(normalized), normalized)

        return normalized
