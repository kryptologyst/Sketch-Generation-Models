"""
Configuration management for sketch generation models.

This module provides:
- YAML-based configuration system
- Model-specific configs
- Training hyperparameters
- Dataset configurations
"""

from typing import Dict, Any, Optional, List, Union
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = 'pix2pix'  # 'pix2pix', 'cyclegan', 'edge_detection'
    in_channels: int = 3
    out_channels: int = 1
    image_size: List[int] = field(default_factory=lambda: [256, 256])
    
    # Pix2Pix specific
    pix2pix_lambda_l1: float = 100.0
    
    # CycleGAN specific
    cyclegan_n_residual_blocks: int = 9
    cyclegan_lambda_cycle: float = 10.0
    cyclegan_lambda_identity: float = 5.0
    
    # Training specific
    gan_mode: str = 'lsgan'  # 'lsgan', 'nsgan', 'wgan'


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    num_epochs: int = 200
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Learning rate scheduling
    lr_scheduler: str = 'linear'  # 'linear', 'cosine', 'step'
    lr_decay_epochs: List[int] = field(default_factory=lambda: [100, 150])
    lr_decay_factor: float = 0.5
    
    # Loss weights
    lambda_l1: float = 100.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    
    # Training stability
    gradient_clip_val: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.999
    
    # Checkpointing
    save_every_n_epochs: int = 10
    save_best_model: bool = True
    early_stopping_patience: int = 20


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_type: str = 'paired'  # 'paired', 'unpaired', 'synthetic'
    data_dir: str = 'data'
    image_dir: str = 'images'
    sketch_dir: str = 'sketches'
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_prob: float = 0.3
    brightness_contrast_prob: float = 0.3
    noise_prob: float = 0.2
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    # Synthetic dataset
    synthetic_num_samples: int = 1000


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    eval_every_n_epochs: int = 5
    num_eval_samples: int = 100
    
    # Metrics
    calculate_fid: bool = True
    calculate_lpips: bool = True
    calculate_edge_consistency: bool = True
    calculate_sketch_quality: bool = True
    
    # Visualization
    save_samples: bool = True
    num_sample_images: int = 8


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = 'logs'
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = 'sketch-generation'
    wandb_entity: Optional[str] = None
    
    # Console logging
    log_every_n_steps: int = 100
    log_images_every_n_epochs: int = 5


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    seed: int = 42
    deterministic: bool = True
    
    # Paths
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'checkpoints'
    sample_dir: str = 'samples'


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = OmegaConf.to_yaml(config)
    
    with open(config_path, 'w') as f:
        f.write(config_dict)


def create_default_configs() -> Dict[str, Config]:
    """Create default configurations for different model types."""
    configs = {}
    
    # Pix2Pix config
    pix2pix_config = Config()
    pix2pix_config.model.model_type = 'pix2pix'
    pix2pix_config.model.pix2pix_lambda_l1 = 100.0
    pix2pix_config.data.dataset_type = 'paired'
    configs['pix2pix'] = pix2pix_config
    
    # CycleGAN config
    cyclegan_config = Config()
    cyclegan_config.model.model_type = 'cyclegan'
    cyclegan_config.model.cyclegan_lambda_cycle = 10.0
    cyclegan_config.model.cyclegan_lambda_identity = 5.0
    cyclegan_config.data.dataset_type = 'unpaired'
    configs['cyclegan'] = cyclegan_config
    
    # Edge detection config
    edge_config = Config()
    edge_config.model.model_type = 'edge_detection'
    edge_config.training.num_epochs = 1  # No training needed
    edge_config.data.dataset_type = 'synthetic'
    configs['edge_detection'] = edge_config
    
    return configs


def get_model_config(config: Config) -> Dict[str, Any]:
    """Extract model-specific configuration."""
    model_config = {
        'model_type': config.model.model_type,
        'in_channels': config.model.in_channels,
        'out_channels': config.model.out_channels,
        'image_size': tuple(config.model.image_size),
        'gan_mode': config.model.gan_mode
    }
    
    if config.model.model_type == 'pix2pix':
        model_config.update({
            'lambda_l1': config.model.pix2pix_lambda_l1
        })
    elif config.model.model_type == 'cyclegan':
        model_config.update({
            'n_residual_blocks': config.model.cyclegan_n_residual_blocks,
            'lambda_cycle': config.model.cyclegan_lambda_cycle,
            'lambda_identity': config.model.cyclegan_lambda_identity
        })
    
    return model_config


def get_training_config(config: Config) -> Dict[str, Any]:
    """Extract training-specific configuration."""
    return {
        'batch_size': config.training.batch_size,
        'num_epochs': config.training.num_epochs,
        'learning_rate': config.training.learning_rate,
        'beta1': config.training.beta1,
        'beta2': config.training.beta2,
        'lambda_l1': config.training.lambda_l1,
        'lambda_cycle': config.training.lambda_cycle,
        'lambda_identity': config.training.lambda_identity,
        'gradient_clip_val': config.training.gradient_clip_val,
        'use_ema': config.training.use_ema,
        'ema_decay': config.training.ema_decay
    }


def get_data_config(config: Config) -> Dict[str, Any]:
    """Extract data-specific configuration."""
    return {
        'dataset_type': config.data.dataset_type,
        'data_dir': config.data.data_dir,
        'image_dir': config.data.image_dir,
        'sketch_dir': config.data.sketch_dir,
        'use_augmentation': config.data.use_augmentation,
        'num_workers': config.data.num_workers,
        'pin_memory': config.data.pin_memory,
        'shuffle_train': config.data.shuffle_train,
        'synthetic_num_samples': config.data.synthetic_num_samples
    }


def validate_config(config: Config) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check model type
    if config.model.model_type not in ['pix2pix', 'cyclegan', 'edge_detection']:
        issues.append(f"Invalid model type: {config.model.model_type}")
    
    # Check dataset type compatibility
    if config.model.model_type == 'pix2pix' and config.data.dataset_type != 'paired':
        issues.append("Pix2Pix requires paired dataset")
    
    if config.model.model_type == 'cyclegan' and config.data.dataset_type != 'unpaired':
        issues.append("CycleGAN requires unpaired dataset")
    
    # Check image size
    if len(config.model.image_size) != 2:
        issues.append("Image size must be a list of 2 integers")
    
    if any(size <= 0 for size in config.model.image_size):
        issues.append("Image size must be positive")
    
    # Check training parameters
    if config.training.batch_size <= 0:
        issues.append("Batch size must be positive")
    
    if config.training.learning_rate <= 0:
        issues.append("Learning rate must be positive")
    
    if config.training.num_epochs <= 0:
        issues.append("Number of epochs must be positive")
    
    return issues
