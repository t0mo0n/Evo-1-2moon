import yaml
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- 配置部分 ----------------
# 假设脚本在 Evo_1/dataset/ 目录下运行，直接读取同级目录的 config.yaml
CONFIG_PATH = "./dataset/config.yaml" 
# ----------------------------------------

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_stats(dataset_path):
    # 直接构建 stats.json 的完整路径 (Linux 绝对路径)
    stats_path = os.path.join(dataset_path, "meta", "stats.json")
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Stats file not found: {stats_path}")
        return None

def collect_data(config):
    records = []
    
    if 'data_groups' not in config:
        print("Error: 'data_groups' not found in config.")
        return pd.DataFrame()

    for group_name, datasets in config['data_groups'].items():
        for dataset_name, dataset_info in datasets.items():
            path = dataset_info.get('path')
            stats = load_stats(path)
            
            if stats:
                # 处理 Action
                if 'action' in stats:
                    mins = np.array(stats['action']['min'])
                    maxs = np.array(stats['action']['max'])
                    ranges = maxs - mins
                    
                    # 记录每个维度的 range
                    for dim_idx, r in enumerate(ranges):
                        records.append({
                            'Group': group_name,
                            'Dataset': dataset_name,
                            'Type': 'Action',
                            'Dimension': dim_idx,
                            'Range': r,
                            'Min': mins[dim_idx],
                            'Max': maxs[dim_idx],
                            'Dim_Count': len(ranges)
                        })
                
                # 处理 State
                if 'observation.state' in stats:
                    mins = np.array(stats['observation.state']['min'])
                    maxs = np.array(stats['observation.state']['max'])
                    ranges = maxs - mins
                    
                    for dim_idx, r in enumerate(ranges):
                        records.append({
                            'Group': group_name,
                            'Dataset': dataset_name,
                            'Type': 'State',
                            'Dimension': dim_idx,
                            'Range': r,
                            'Min': mins[dim_idx],
                            'Max': maxs[dim_idx],
                            'Dim_Count': len(ranges)
                        })
    
    return pd.DataFrame(records)

def visualize_heatmap(df, data_type='Action'):
    """
    绘制热力图：展示每个数据集在每个维度的 Range 大小。
    这能最直观地看到数据的“形状”和“尺度”。
    """
    subset = df[df['Type'] == data_type]
    if subset.empty:
        return

    # Pivot table: Index=Dataset, Columns=Dimension, Values=Range
    pivot_df = subset.pivot_table(index='Dataset', columns='Dimension', values='Range')
    
    # 根据 Range 的均值对数据集进行排序，方便观察
    pivot_df['mean_range'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('mean_range', ascending=False)
    pivot_df = pivot_df.drop(columns=['mean_range'])

    plt.figure(figsize=(20, max(10, len(pivot_df) * 0.3)))
    
    # 使用 Robust 的配色，因为可能有极值
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis", 
                cbar_kws={'label': 'Range Magnitude (Max - Min)'})
    
    plt.title(f'{data_type} Space Range Heatmap (Sorted by Mean Range)\nIdentifying Scale & Dimensionality difference')
    plt.tight_layout()
    plt.show()

def visualize_distribution_histogram(df, data_type='Action'):
    """
    绘制直方图：你想要的可视化。
    展示 Range 长度的频数分布。
    """
    subset = df[df['Type'] == data_type]
    if subset.empty:
        return

    plt.figure(figsize=(12, 6))
    
    # 使用 Log scale x轴可能更好，因为不同数据集尺度差异可能很大（例如 delta action vs absolute pose）
    sns.histplot(data=subset, x="Range", hue="Group", element="step", bins=50, log_scale=(True, False))
    
    plt.title(f'Distribution of {data_type} Range Lengths (Log Scale X-Axis)')
    plt.xlabel('Range Length (Max - Min)')
    plt.ylabel('Count of Dimensions across all datasets')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

def visualize_rms_ranking(df, data_type='Action'):
    """
    计算 RMS (Root Mean Square) 并排序，作为对数据集尺度的一种综合评分。
    RMS = sqrt(mean(range^2))
    这比单纯的向量模长更公平，不受维度数量影响。
    """
    subset = df[df['Type'] == data_type]
    if subset.empty:
        return

    # 计算 RMS
    def calc_rms(x):
        return np.sqrt(np.mean(np.square(x)))

    agg = subset.groupby(['Group', 'Dataset'])['Range'].agg(calc_rms).reset_index()
    agg = agg.rename(columns={'Range': 'RMS_Range'})
    agg = agg.sort_values('RMS_Range', ascending=False)

    plt.figure(figsize=(14, 8))
    sns.barplot(data=agg, x='RMS_Range', y='Dataset', hue='Group', dodge=False)
    plt.title(f'{data_type} Dataset Scale Ranking (Metric: RMS of Ranges)')
    plt.xlabel('Root Mean Square of Ranges (Scale Indicator)')
    plt.tight_layout()
    plt.show()
    
def visualize_per_dimension_ranges(df, data_type='Action'):
    """
    为每个维度绘制一张图。
    每张图展示所有含有该维度的数据集在该维度上的绝对取值范围 [Min, Max]。
    按照 (Min+Max)/2 的均值进行排序。
    """
    subset = df[df['Type'] == data_type]
    if subset.empty:
        return

    # 获取所有出现的维度
    all_dims = sorted(subset['Dimension'].unique())

    print(f"Generating range plots for {len(all_dims)} dimensions in {data_type} space...")

    for dim in all_dims:
        # 筛选当前维度的数据
        dim_data = subset[subset['Dimension'] == dim].copy()
        if dim_data.empty:
            continue
        
        # 计算中心点用于排序：(Min + Max) / 2
        dim_data['Center'] = (dim_data['Min'] + dim_data['Max']) / 2
        # 排序
        dim_data = dim_data.sort_values('Center')
        
        # 动态调整图片高度，取决于数据集数量
        n_datasets = len(dim_data)
        fig_height = max(6, n_datasets * 0.25)
        
        plt.figure(figsize=(12, fig_height))
        
        # 绘制水平线段 (Min 到 Max)
        # hlines(y, xmin, xmax)
        plt.hlines(y=range(n_datasets), xmin=dim_data['Min'], xmax=dim_data['Max'], 
                   color='skyblue', alpha=0.8, linewidth=2)
        
        # 绘制 Min 和 Max 的端点，以及 Center 点
        plt.scatter(dim_data['Min'], range(n_datasets), color='green', s=15, marker='|', label='Min')
        plt.scatter(dim_data['Max'], range(n_datasets), color='red', s=15, marker='|', label='Max')
        plt.scatter(dim_data['Center'], range(n_datasets), color='blue', s=10, marker='o', label='Center')

        # 设置Y轴标签为数据集名称
        plt.yticks(range(n_datasets), dim_data['Dataset'], fontsize=9)
        
        # 添加装饰
        plt.title(f'{data_type} Dimension {dim}: Absolute Value Range (Min to Max)', fontsize=14)
        plt.xlabel('Actual Value', fontsize=12)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # 仅在第一张图中显示图例，避免遮挡
        if dim == all_dims[0]:
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found: {CONFIG_PATH}")
        return

    config = load_yaml(CONFIG_PATH)
    print("Loading datasets stats...")
    df = collect_data(config)
    
    if df.empty:
        print("No valid stats data found found. Please check paths.")
        return

    print(f"Loaded records for {len(df['Dataset'].unique())} datasets.")

    # 1. Action Space Heatmap (最推荐，用于分组)
    print("Generating Action Heatmap...")
    visualize_heatmap(df, 'Action')
    
    # 2. State Space Heatmap
    print("Generating State Heatmap...")
    visualize_heatmap(df, 'State')

    # 3. Overall Distribution (你想要的直方图)
    print("Generating Distribution Histogram...")
    visualize_distribution_histogram(df, 'Action')

    # 4. RMS Ranking (更公平的“模长”替代方案)
    print("Generating RMS Ranking...")
    visualize_rms_ranking(df, 'Action')
    
    print("Generating Per-Dimension Absolute Range Plots...")
    visualize_per_dimension_ranges(df, 'Action')
    visualize_per_dimension_ranges(df, 'State')

if __name__ == "__main__":
    main()