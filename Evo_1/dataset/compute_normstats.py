"""Utility to peek Lerobot v2.1 datasets and inspect normalization stats."""

from __future__ import annotations

import argparse
import json
import logging
import yaml
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyarrow.parquet as pq

REQUIRED_METRICS: Tuple[str, ...] = ("std", "mean", "min", "max", "q01", "q99")


def compute_normstats(dataset_path: Path, use_delta_actions: bool = True, action_horizon: int = 50) -> Dict[str, object]:
    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data"
    info_path = meta_dir / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing meta/info.json under {dataset_path}")

    info = json.loads(info_path.read_text())
    print(f"Inspecting dataset at {dataset_path}")
    print(
        "Meta summary:"
        f" {info.get('codebase_version', 'unknown')} |"
        f" episodes={info.get('total_episodes', 'n/a')} |"
        f" frames={info.get('total_frames', 'n/a')}"
    )

    parquet_files: List[Path] = []
    for chunk in sorted(data_dir.glob("chunk-*/")):
        parquet_files.extend(sorted(chunk.glob("episode_*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files located under {data_dir}")

    total_frames = 0
    per_episode: Dict[int, int] = {}
    action_dim = None
    state_dim = None
    action_batches: List[np.ndarray] = []
    state_batches: List[np.ndarray] = []
    
    print(f"Processing {len(parquet_files)} episodes...")
    if use_delta_actions:
        print(f"Computing relative action statistics with horizon {action_horizon}...")
    else:
        print("Computing absolute action statistics...")

    for pq_path in parquet_files:
        table = pq.read_table(pq_path, columns=["action", "observation.state"])
        frames_here = table.num_rows
        total_frames += frames_here

        try:
            episode_idx = int(pq_path.stem.split("_")[-1])
        except ValueError:
            episode_idx = len(per_episode)
        per_episode[episode_idx] = frames_here

        if frames_here:
            actions_np = _column_to_numpy(table.column("action"))
            states_np = _column_to_numpy(table.column("observation.state"))
            
            if use_delta_actions:
                N, D = actions_np.shape
                
                pad_vec = actions_np[-1:] # (1, D)
                padding = np.repeat(pad_vec, action_horizon, axis=0) # (H, D)
                actions_padded = np.concatenate([actions_np, padding], axis=0) # (N+H, D)
                
                # Create sliding windows
                windows_view = sliding_window_view(actions_padded, action_horizon, axis=0)
                windows_view = windows_view[:N] # (N, D, H)
                windows = windows_view.transpose(0, 2, 1) # (N, H, D)
                
                min_dim = min(states_np.shape[1], D)
                state_broadcast = np.zeros((N, 1, D), dtype=actions_np.dtype)
                state_broadcast[:, 0, :min_dim] = states_np[:, :min_dim]
                
                # Compute relative actions
                relative_actions = windows - state_broadcast
                actions_to_store = relative_actions.reshape(-1, D) # (N*H, D)
                action_batches.append(actions_to_store)
            else:
                action_batches.append(actions_np)
                
            state_batches.append(states_np)
            if action_dim is None:
                action_dim = actions_np.shape[1]
            if state_dim is None:
                state_dim = states_np.shape[1]

    lengths = list(per_episode.values())
    length_stats = {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(mean(lengths), 2),
    }

    print(
        f"Loaded {len(per_episode)} episodes"
        f" with {total_frames} frames in action/state columns."
    )
    print(
        f"Action rows: {sum(len(b) for b in action_batches)} | dim={action_dim}"
        f"\nState rows:  {sum(len(b) for b in state_batches)} | dim={state_dim}"
    )
    print(
        "Episode length stats (frames):"
        f" min={length_stats['min']}"
        f" max={length_stats['max']}"
        f" mean={length_stats['avg']}"
    )

    actions = _concat_batches(action_batches)
    states = _concat_batches(state_batches)
    stats_payload = {
        "action": _vector_stats(actions),
        "observation.state": _vector_stats(states),
    }

    stats_path = meta_dir / "stats.json"
    stats_path.write_text(
        json.dumps(stats_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote global stats (includes q01/q99) to {stats_path}.")

    # Diagnose existing per-episode stats coverage for context.
    per_episode_stats = meta_dir / "episodes_stats.jsonl"
    present_metrics: Iterable[str] = tuple()
    missing_metrics: Iterable[str] = REQUIRED_METRICS
    if per_episode_stats.exists():
        present_metrics, missing_metrics = _collect_metric_keys(per_episode_stats)
        print(
            f"Per-episode stats file: {per_episode_stats.name}."
            " Metrics present: " + ", ".join(present_metrics)
        )
        if missing_metrics:
            print(
                "Missing targets (per-episode file): "
                + ", ".join(
                    metric for metric in REQUIRED_METRICS if metric in missing_metrics
                )
            )
        else:
            print("Per-episode stats already cover all required metrics.")
    else:
        print("episodes_stats.jsonl not found; only global stats were produced.")

    return {
        "dataset_root": str(dataset_path),
        "episodes_found": len(per_episode),
        "frames_read": total_frames,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "episode_length_stats": length_stats,
        "stats_file": stats_path.name,
        "metrics_present": list(present_metrics),
        "metrics_missing": list(missing_metrics),
    }


def _concat_batches(batches: List[np.ndarray]) -> np.ndarray:
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0)


def _column_to_numpy(column) -> np.ndarray:
    data = column.to_pylist()
    if not data:
        return np.empty((0, 0), dtype=np.float32)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array


def _vector_stats(array: np.ndarray) -> Dict[str, List[float]]:
    if array.size == 0:
        return {metric: [] for metric in REQUIRED_METRICS}

    arr = array.astype(np.float64, copy=False)
    stats = {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0, ddof=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }
    return stats


def _collect_metric_keys(stats_path: Path) -> Tuple[List[str], List[str]]:
    present = set()
    with stats_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            stats_section = payload.get("stats")
            if not isinstance(stats_section, dict):
                continue

            if stats_section and all(
                isinstance(v, dict) for v in stats_section.values()
            ):
                stats_dicts = stats_section.values()
            else:
                stats_dicts = [stats_section]

            for entry in stats_dicts:
                if not isinstance(entry, dict):
                    continue
                for key in entry.keys():
                    if key in REQUIRED_METRICS:
                        present.add(key)
            if len(present) == len(REQUIRED_METRICS):
                break

    missing = [metric for metric in REQUIRED_METRICS if metric not in present]
    return sorted(present), missing


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect lerobot dataset stats.")
    parser.add_argument(
        "config_path",
        type=str,
        help="指向数据集配置文件（如 config.yaml）的路径。",
    )
    parser.add_argument(
        "--use_delta_actions",
        type=str2bool,
        default=True,
        help="使用相对于初始状态的相对动作的统计数据。",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=50,
        help="动作序列的长度 (action_horizon)。"
    )
    args = parser.parse_args()
    config_path = Path(args.config_path)
    if not config_path.exists():
        logging.error(f"配置文件不存在: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for arm_name, arm_config in config.get('data_groups', {}).items():
        for dataset_name, dataset_config in arm_config.items():
            path_str = dataset_config.get('path')
            if not path_str:
                logging.warning(f"数据集 '{arm_name}/{dataset_name}' 未配置路径，跳过。")
                continue
            
            dataset_path = Path(path_str)
            compute_normstats(dataset_path, use_delta_actions=args.use_delta_actions, action_horizon=args.action_horizon)


if __name__ == "__main__":
    main()
