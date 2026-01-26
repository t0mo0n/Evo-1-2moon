# extandable dataset processing suites for various robotics datasets

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class ProcessedData:
    """处理后的数据结构"""
    state: Optional[np.ndarray]  # 初始状态 (用于归一化参考)
    actions: np.ndarray          # action 序列 [horizon, action_dim]
    state_dim: int               # 状态维度
    action_dim: int              # 动作维度


class BaseProcessSuite(ABC):
    """处理套组基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 套组配置，可包含特定套组需要的额外参数
        """
        self.config = config or {}
    
    @abstractmethod
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        """从单行数据中提取状态"""
        pass
    
    @abstractmethod
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        """从子数据帧中提取动作序列"""
        pass
    
    def compute_relative_actions(
        self, 
        actions: np.ndarray, 
        states: np.ndarray,
        relative_dims: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        计算相对动作
        
        Args:
            actions: 原始动作 [horizon, action_dim]
            states: 状态序列 [horizon, state_dim]
            relative_dims: 需要转换为相对值的维度范围列表，如 [(0, 6), (7, 13)]
        
        Returns:
            相对动作 [horizon, action_dim]
        """
        relative_actions = actions.copy()
        
        for start, end in relative_dims:
            if actions.shape[1] >= end and states.shape[1] >= end:
                relative_actions[:, start:end] = actions[:, start:end] - states[:, start:end]
        
        return relative_actions
    
    def process(
        self, 
        sub_df: pd.DataFrame, 
        use_delta_action: bool = True
    ) -> ProcessedData:
        """
        主处理方法
        
        Args:
            sub_df: 包含 action_horizon 行的子数据帧
            use_delta_action: 是否转换为相对动作
        
        Returns:
            ProcessedData 对象
        """
        # 提取初始状态
        init_state = self.extract_state(sub_df.iloc[0])
        
        # 提取动作序列
        actions = self.extract_actions(sub_df)
        
        # 如果需要相对动作，进行转换
        if use_delta_action and init_state is not None:
            actions = self._convert_to_relative(sub_df, actions)
        
        return ProcessedData(
            state=init_state,
            actions=actions.tolist() if isinstance(actions, np.ndarray) else actions,
            state_dim=len(init_state) if init_state is not None else 0,
            action_dim=actions.shape[1] if isinstance(actions, np.ndarray) else len(actions[0])
        )
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        """转换为相对动作（子类可重写以自定义行为）"""
        return actions



class DefaultSuite(BaseProcessSuite):
    """
    默认套组：直接读取 observation.state 和 action
    适用于大多数标准 LeRobot 数据集
    """
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get("observation.state", None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df["action"].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        states = np.stack(sub_df["observation.state"].to_list())
        
        # 默认：前6维为末端执行器位姿，7-13维为第二臂（如有）
        relative_dims = []
        action_dim = actions.shape[1]
        state_dim = states.shape[1]
        
        if action_dim >= 6 and state_dim >= 6:
            relative_dims.append((0, 6))
        if action_dim >= 13 and state_dim >= 13:
            relative_dims.append((7, 13))
        
        return self.compute_relative_actions(actions, states, relative_dims)


class FrankaEEPoseSuite(BaseProcessSuite):
    """
    Franka 末端执行器位姿套组
    适用于: LIBERO, austin_buds, austin_sirius 等
    
    特点:
    - State: observation.state (7D: xyz + rpy + gripper)
    - Action: action (通常是 7D: delta_xyz + delta_rpy + gripper)
    - Gripper 保持绝对值
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.gripper_indices = self.config.get("gripper_indices", [6])  # gripper 所在的索引
        self.ee_pose_dims = self.config.get("ee_pose_dims", (0, 6))  # 末端执行器位姿维度
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            logging.warning(f"Cannot extract states from {self.state_key}, skipping relative conversion")
            return actions
        
        relative_actions = actions.copy()
        start, end = self.ee_pose_dims
        
        if actions.shape[1] >= end and states.shape[1] >= end:
            relative_actions[:, start:end] = actions[:, start:end] - states[:, start:end]
        
        # Gripper 保持绝对值（不做转换）
        
        return relative_actions


class FrankaJointAngleSuite(BaseProcessSuite):
    """
    Franka 关节角度套组
    适用于: RLBench, berkeley_rpt 等
    
    特点:
    - State: observation.state (8D: 7D 关节角度 + gripper)
    - Action: action (8D: 7D 关节角度 + gripper)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.joint_dims = self.config.get("joint_dims", 7)
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        # 关节角度：全部维度都转换为相对值（除了 gripper）
        relative_dims = [(0, min(self.joint_dims, actions.shape[1], states.shape[1]))]
        return self.compute_relative_actions(actions, states, relative_dims)


class DroidEEFSuite(BaseProcessSuite):
    """
    DROID 末端执行器套组
    适用于: droid_101_eef
    
    特点:
    - State: observation.state.cartesian_position (6D) + gripper from state[6]
    - Action: action.cartesian_position (6D) + gripper from action[6]
    - gripper反向, 需要手动拼接 gripper
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state.cartesian_position")
        self.action_key = self.config.get("action_key", "action.cartesian_position")
        self.action_gripper_key = self.config.get("action_gripper_key", "action.gripper_position")
        self.state_gripper_key = self.config.get("state_gripper_key", "observation.state.gripper_position")
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state_pos = row.get(self.state_key, None)
        state_gripper_raw = row.get(self.state_gripper_key, None)
        if state_pos is not None and state_gripper_raw is not None:
            state_gripper = 1 - state_gripper_raw
            return np.concatenate([np.array(state_pos), np.array([state_gripper])], axis=0)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        cart_actions = np.stack(sub_df[self.action_key].to_list())
        
        gripper_raw = sub_df[self.action_gripper_key].to_list()
        gripper = 1 - np.array(gripper_raw).reshape(-1, 1)

        return np.concatenate([cart_actions, gripper], axis=1)
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        relative_actions = actions.copy()
        # 只有前6维（cartesian position）转换为相对值
        if actions.shape[1] >= 6 and states.shape[1] >= 6:
            relative_actions[:, :6] = actions[:, :6] - states[:, :6]
        
        return relative_actions


class AlohaJointAngleSuite(BaseProcessSuite):
    """
    ALOHA 双臂关节角度套组
    适用于: RoboTwin, ALOHA 数据集
    
    特点:
    - State: observation.state (14D: 7D 左臂 + 7D 右臂)
    - Action: action (14D: 7D 左臂 + 7D 右臂)
    - 索引 6 和 13 是 gripper，保持绝对值
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.arm_dims = self.config.get("arm_dims", 7)  # 每条臂的维度
        self.gripper_indices = self.config.get("gripper_indices", [6, 13])
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        relative_actions = actions.copy()
        action_dim = actions.shape[1]
        state_dim = states.shape[1]
        
        # 左臂关节 (0-5)，跳过 gripper (6)
        if action_dim >= 6 and state_dim >= 6:
            relative_actions[:, :6] = actions[:, :6] - states[:, :6]
        
        # 右臂关节 (7-12)，跳过 gripper (13)
        if action_dim >= 13 and state_dim >= 13:
            relative_actions[:, 7:13] = actions[:, 7:13] - states[:, 7:13]
        
        return relative_actions


class MetaWorldSuite(BaseProcessSuite):
    """
    MetaWorld 数据集套组
    
    特点:
    - State: observation.state
    - Action: action (4D: delta xyz + gripper)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        # MetaWorld 的 action 本身就是增量，通常不需要额外处理
        return actions


class CustomSuite(BaseProcessSuite):
    """
    自定义套组：通过配置文件完全自定义行为
    
    配置示例:
    {
        "state_key": "observation.state",
        "action_key": "action",
        "action_concat_keys": ["action.cartesian", "action.gripper"],  # 可选：拼接多个键
        "relative_dims": [[0, 6], [7, 13]],  # 可选：需要转换为相对值的维度
        "gripper_indices": [6, 13]  # 可选：gripper 索引（保持绝对值）
    }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.action_concat_keys = self.config.get("action_concat_keys", None)
        self.relative_dims = self.config.get("relative_dims", [[0, 6]])
        self.gripper_indices = self.config.get("gripper_indices", [])
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        if self.action_concat_keys:
            # 拼接多个键
            arrays = []
            for key in self.action_concat_keys:
                arr = np.stack(sub_df[key].to_list())
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            return np.concatenate(arrays, axis=1)
        else:
            return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        relative_dims = [tuple(d) for d in self.relative_dims]
        return self.compute_relative_actions(actions, states, relative_dims)



# suite registry
SUITE_REGISTRY: Dict[str, type] = {
    "default": DefaultSuite,
    "franka_ee_pose": FrankaEEPoseSuite,
    "franka_joint_angle": FrankaJointAngleSuite,
    "droid_eef": DroidEEFSuite,
    "aloha_joint_angle": AlohaJointAngleSuite,
    "metaworld": MetaWorldSuite,
    "custom": CustomSuite,
}


def register_suite(name: str, suite_class: type):
    """Register a new suite"""
    if not issubclass(suite_class, BaseProcessSuite):
        raise TypeError(f"Suite class must inherit from BaseProcessSuite")
    SUITE_REGISTRY[name] = suite_class


def get_suite(name: str, config: Dict[str, Any] = None) -> BaseProcessSuite:
    """
    获取套组实例
    
    Args:
        name: 套组名称
        config: 套组配置
    
    Returns:
        套组实例
    """
    if name not in SUITE_REGISTRY:
        available = list(SUITE_REGISTRY.keys())
        raise ValueError(f"Unknown suite '{name}'. Available suites: {available}")
    
    return SUITE_REGISTRY[name](config)


def list_available_suites() -> List[str]:
    """列出所有可用的套组"""
    return list(SUITE_REGISTRY.keys())