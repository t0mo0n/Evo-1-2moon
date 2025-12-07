import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_relative_action_stats_for_episode(parquet_path: Path, action_horizon: int) -> dict:
    """
    加载单个episode的parquet文件，精确计算所有滑动窗口的相对动作的统计数据。

    Args:
        parquet_path (Path): 指向单个parquet文件的路径。
        action_horizon (int): 动作序列的长度。

    Returns:
        dict: 包含该episode相对动作统计信息的字典。
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        last_row = df.iloc[-1:]  
        padding_rows = pd.concat([last_row] * action_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)

        if 'action' not in df.columns or 'observation.state' not in df.columns:
            logging.warning(f"列 'action' 或 'observation.state' 在文件 {parquet_path} 中未找到，跳过。")
            return None

        all_relative_actions = []
        
        # 模拟数据加载时的滑动窗口
        for i in range(len(df) - action_horizon + 1):
            sub_df = df.iloc[i : i + action_horizon]
            
            actions = np.stack(sub_df["action"].to_list())
            init_state = df.iloc[i].get("observation.state") # 使用i而不是sub_df.iloc[0]以获取原始df中的状态

            if init_state is not None:
                min_dim = min(len(init_state), actions.shape[1])
                
                init_state_broadcast = np.zeros_like(actions)
                init_state_broadcast[:, :min_dim] = init_state[:min_dim]
                
                relative_actions = actions - init_state_broadcast
                all_relative_actions.append(relative_actions)
            else:
                logging.warning(f"文件 {parquet_path} 中索引 {i} 处缺少状态，无法计算相对动作。")
                # 如果没有状态，则无法计算相对动作，可以跳过或记录绝对动作
                all_relative_actions.append(actions)

        if not all_relative_actions:
            return None

        # 将所有窗口的相对动作合并成一个大数组
        # all_relative_actions 是一个列表，其中每个元素是 (action_horizon, action_dim) 的数组
        # 我们需要将它们堆叠成 (N * action_horizon, action_dim)
        all_actions_np = np.concatenate(all_relative_actions, axis=0)

        stats = {
            "min": np.min(all_actions_np, axis=0).tolist(),
            "max": np.max(all_actions_np, axis=0).tolist(),
            "mean": np.mean(all_actions_np, axis=0).tolist(),
            "std": np.std(all_actions_np, axis=0).tolist(),
        }

        return {
            "episode_id": parquet_path.stem,
            "stats": {
                "action_relative": stats
            }
        }
    except Exception as e:
        logging.error(f"处理文件 {parquet_path} 时出错: {e}")
        return None

def process_dataset(dataset_path: Path, action_horizon: int):
    """
    处理单个数据集目录中的所有parquet文件,并生成统计文件。

    Args:
        dataset_path (Path): 数据集根目录的路径。
        action_horizon (int): 动作序列的长度。
    """
    data_dir = dataset_path / "data"
    meta_dir = dataset_path / "meta"
    output_path = meta_dir / "episodes_stats_action_relative.jsonl"

    if not data_dir.exists():
        logging.warning(f"数据目录 {data_dir} 不存在，跳过数据集 {dataset_path.name}。")
        return

    meta_dir.mkdir(exist_ok=True)

    parquet_files = sorted(list(data_dir.glob("*/*.parquet")))
    if not parquet_files:
        logging.warning(f"在 {data_dir} 中未找到Parquet文件，跳过。")
        return

    logging.info(f"正在处理数据集 '{dataset_path.name}' 中的 {len(parquet_files)} 个 episodes...")

    with open(output_path, 'w') as f:
        for parquet_file in tqdm(parquet_files, desc=f"计算 {dataset_path.name} 的统计数据"):
            episode_stats = compute_relative_action_stats_for_episode(parquet_file, action_horizon)
            if episode_stats:
                f.write(json.dumps(episode_stats) + '\n')

    logging.info(f"相对动作统计数据已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="为LeRobot数据集计算相对于初始状态的相对动作的统计数据。")
    parser.add_argument(
        "config_path",
        type=str,
        help="指向数据集配置文件（如 config.yaml）的路径。",
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
            process_dataset(dataset_path, args.action_horizon)

if __name__ == "__main__":
    main()