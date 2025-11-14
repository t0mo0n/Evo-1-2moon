from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("evo1")
@dataclass
class Evo1Config(PreTrainedConfig):

    n_obs_steps: int = 1
    chunk_size: int = 50              
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 24
    max_action_dim: int = 24

    resize_imgs_with_padding: tuple[int, int] = (448, 448)
    empty_cameras: int = 0

    # === VLM backbone ===
    vlm_model_name: str = "OpenGVLab/InternVL3-1B"
    load_vlm_weights: bool = True
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = 0
    pad_language_to: str = "max_length"

    # === Model head ===
    action_head: str = "flowmatching"
    dropout: float = 0.2
    num_layers: int = 8
    per_action_dim: int = 24
    state_dim: int = 24
    action_dim: int = 1200

    # === Training hyperparameters ===
    device: str = "cuda"
    use_amp: bool = False
    use_augmentation: bool = True
    binarize_gripper: bool = False
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # === Optimizer ===
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-3
    optimizer_grad_clip_norm: float = 1.0

    # === Scheduler ===
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    # === Training meta ===
    batch_size: int = 16
    max_steps: int = 30000
    log_interval: int = 10
    ckpt_interval: int = 2500
    save_dir: str = "./checkpoints/evo1"
    resume: bool = True
    resume_path: str = ""
    resume_pretrain: bool = True

    # === Other ===
    num_workers: int = 4
    disable_wandb: bool = True
    return_cls_only: bool = False
    run_name: str = "Evo1_Default"
    dataset_type: str = "lerobot"
    dataset_config_path: str = ""
    data_paths: str | None = None

    # === Diffusion / Decoding ===
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    def __post_init__(self):
        super().__post_init__()
        # Basic sanity check
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be ≤ chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        """Add placeholder cameras if needed."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
