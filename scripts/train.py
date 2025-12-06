#!/usr/bin/env python3
"""
Main training script for sketch generation models.

This script provides a unified interface for training different types of
sketch generation models (Pix2Pix, CycleGAN, Edge Detection).
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import (
    PairedImageSketchDataset, 
    UnpairedImageSketchDataset, 
    SyntheticSketchDataset,
    get_transforms,
    create_dataloader,
    create_synthetic_dataset
)
from src.utils import Pix2PixTrainer, CycleGANTrainer, save_checkpoint, create_sample_grid
from src.utils.metrics import ModelEvaluator
from src.configs import (
    Config, 
    load_config, 
    save_config, 
    create_default_configs,
    get_model_config,
    get_training_config,
    get_data_config,
    validate_config
)


def setup_device(config: Config) -> torch.device:
    """Setup device based on configuration."""
    if config.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    return device


def setup_dataloaders(config: Config) -> tuple:
    """Setup data loaders based on configuration."""
    data_config = get_data_config(config)
    
    # Create synthetic dataset if needed
    if data_config['dataset_type'] == 'synthetic':
        data_dir = Path(data_config['data_dir'])
        if not data_dir.exists():
            print("Creating synthetic dataset...")
            create_synthetic_dataset(
                str(data_dir),
                data_config['synthetic_num_samples'],
                tuple(config.model.image_size)
            )
    
    # Setup transforms
    transform = get_transforms(
        tuple(config.model.image_size),
        mode='train',
        use_augmentation=data_config['use_augmentation']
    )
    
    # Create datasets
    if data_config['dataset_type'] == 'paired':
        train_dataset = PairedImageSketchDataset(
            data_config['image_dir'],
            data_config['sketch_dir'],
            transform=transform,
            image_size=tuple(config.model.image_size),
            mode='train'
        )
        val_dataset = PairedImageSketchDataset(
            data_config['image_dir'],
            data_config['sketch_dir'],
            transform=None,
            image_size=tuple(config.model.image_size),
            mode='val'
        )
    elif data_config['dataset_type'] == 'unpaired':
        train_dataset = UnpairedImageSketchDataset(
            data_config['image_dir'],
            data_config['sketch_dir'],
            transform=transform,
            image_size=tuple(config.model.image_size),
            mode='train'
        )
        val_dataset = UnpairedImageSketchDataset(
            data_config['image_dir'],
            data_config['sketch_dir'],
            transform=None,
            image_size=tuple(config.model.image_size),
            mode='val'
        )
    else:  # synthetic
        train_dataset = SyntheticSketchDataset(
            data_config['synthetic_num_samples'],
            tuple(config.model.image_size),
            mode='train'
        )
        val_dataset = SyntheticSketchDataset(
            data_config['synthetic_num_samples'],
            tuple(config.model.image_size),
            mode='val'
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle_train'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    return train_loader, val_loader


def setup_models(config: Config, device: torch.device) -> tuple:
    """Setup models based on configuration."""
    model_config = get_model_config(config)
    
    if config.model.model_type == 'pix2pix':
        generator = create_model('pix2pix_generator', **model_config)
        discriminator = create_model('pix2pix_discriminator', **model_config)
        return generator, discriminator, None, None
    
    elif config.model.model_type == 'cyclegan':
        g_image_to_sketch = create_model('cyclegan_generator', **model_config)
        g_sketch_to_image = create_model('cyclegan_generator', **model_config)
        d_image = create_model('cyclegan_discriminator', **model_config)
        d_sketch = create_model('cyclegan_discriminator', **model_config)
        return g_image_to_sketch, g_sketch_to_image, d_image, d_sketch
    
    elif config.model.model_type == 'edge_detection':
        model = create_model('edge_detection')
        return model, None, None, None
    
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")


def train_pix2pix(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device
) -> None:
    """Train Pix2Pix model."""
    training_config = get_training_config(config)
    
    # Setup trainer
    trainer = Pix2PixTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr=training_config['learning_rate'],
        beta1=training_config['beta1'],
        lambda_l1=training_config['lambda_l1'],
        gan_mode=config.model.gan_mode
    )
    
    # Setup evaluator
    evaluator = ModelEvaluator(device)
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(training_config['num_epochs']):
        # Training
        generator.train()
        discriminator.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['num_epochs']}")
        
        for batch in pbar:
            losses = trainer.train_step(batch)
            epoch_losses.append(losses)
            
            pbar.set_postfix({
                'D Loss': f"{losses['d_loss']:.4f}",
                'G Loss': f"{losses['g_loss']:.4f}"
            })
        
        # Average losses
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}
        train_losses.append(avg_losses)
        
        # Validation
        if epoch % config.evaluation.eval_every_n_epochs == 0:
            val_results = evaluator.evaluate_model(generator, val_loader, config.evaluation.num_eval_samples)
            val_losses.append(val_results)
            
            print(f"Epoch {epoch+1} - Val FID: {val_results['fid']:.4f}, LPIPS: {val_results['lpips']:.4f}")
            
            # Save best model
            if val_results['fid'] < best_loss:
                best_loss = val_results['fid']
                save_checkpoint(
                    generator,
                    trainer.optimizer_g,
                    epoch,
                    val_results['fid'],
                    f"{config.checkpoint_dir}/best_generator.pth"
                )
                save_checkpoint(
                    discriminator,
                    trainer.optimizer_d,
                    epoch,
                    val_results['fid'],
                    f"{config.checkpoint_dir}/best_discriminator.pth"
                )
        
        # Save samples
        if epoch % config.evaluation.log_images_every_n_epochs == 0:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                fake_sketches = trainer.generate_samples(sample_batch)
                
                fig = create_sample_grid(
                    sample_batch['image'],
                    sample_batch['sketch'],
                    fake_sketches,
                    config.evaluation.num_sample_images
                )
                
                fig.savefig(f"{config.sample_dir}/samples_epoch_{epoch+1}.png")
                plt.close(fig)
        
        # Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0:
            save_checkpoint(
                generator,
                trainer.optimizer_g,
                epoch,
                avg_losses['g_loss'],
                f"{config.checkpoint_dir}/generator_epoch_{epoch+1}.pth"
            )
            save_checkpoint(
                discriminator,
                trainer.optimizer_d,
                epoch,
                avg_losses['d_loss'],
                f"{config.checkpoint_dir}/discriminator_epoch_{epoch+1}.pth"
            )


def train_cyclegan(
    g_image_to_sketch: nn.Module,
    g_sketch_to_image: nn.Module,
    d_image: nn.Module,
    d_sketch: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device
) -> None:
    """Train CycleGAN model."""
    training_config = get_training_config(config)
    
    # Setup trainer
    trainer = CycleGANTrainer(
        g_image_to_sketch=g_image_to_sketch,
        g_sketch_to_image=g_sketch_to_image,
        d_image=d_image,
        d_sketch=d_sketch,
        device=device,
        lr=training_config['learning_rate'],
        beta1=training_config['beta1'],
        lambda_cycle=training_config['lambda_cycle'],
        lambda_identity=training_config['lambda_identity'],
        gan_mode=config.model.gan_mode
    )
    
    # Setup evaluator
    evaluator = ModelEvaluator(device)
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(training_config['num_epochs']):
        # Training
        g_image_to_sketch.train()
        g_sketch_to_image.train()
        d_image.train()
        d_sketch.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['num_epochs']}")
        
        for batch_image, batch_sketch in zip(train_loader, train_loader):
            losses = trainer.train_step(batch_image, batch_sketch)
            epoch_losses.append(losses)
            
            pbar.set_postfix({
                'G Loss': f"{losses['g_loss']:.4f}",
                'D Loss': f"{losses['d_loss']:.4f}"
            })
        
        # Average losses
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}
        train_losses.append(avg_losses)
        
        # Validation
        if epoch % config.evaluation.eval_every_n_epochs == 0:
            val_results = evaluator.evaluate_model(g_image_to_sketch, val_loader, config.evaluation.num_eval_samples)
            val_losses.append(val_results)
            
            print(f"Epoch {epoch+1} - Val FID: {val_results['fid']:.4f}, LPIPS: {val_results['lpips']:.4f}")
            
            # Save best model
            if val_results['fid'] < best_loss:
                best_loss = val_results['fid']
                save_checkpoint(
                    g_image_to_sketch,
                    trainer.optimizer_g,
                    epoch,
                    val_results['fid'],
                    f"{config.checkpoint_dir}/best_g_image_to_sketch.pth"
                )
        
        # Save samples
        if epoch % config.evaluation.log_images_every_n_epochs == 0:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                samples = trainer.generate_samples(sample_batch)
                
                fig = create_sample_grid(
                    sample_batch['image'],
                    sample_batch['sketch'],
                    samples['fake_sketches'],
                    config.evaluation.num_sample_images
                )
                
                fig.savefig(f"{config.sample_dir}/samples_epoch_{epoch+1}.png")
                plt.close(fig)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train sketch generation models')
    parser.add_argument('--config', type=str, default='configs/pix2pix.yaml', help='Path to config file')
    parser.add_argument('--model-type', type=str, choices=['pix2pix', 'cyclegan', 'edge_detection'], 
                       help='Override model type')
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        # Use default config
        configs = create_default_configs()
        model_type = args.model_type or 'pix2pix'
        config = configs[model_type]
    
    # Override config with command line arguments
    if args.model_type:
        config.model.model_type = args.model_type
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        return
    
    # Setup directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.sample_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(config)
    
    # Setup data loaders
    train_loader, val_loader = setup_dataloaders(config)
    
    # Setup models
    models = setup_models(config, device)
    
    print(f"Training {config.model.model_type} model...")
    print(f"Configuration: {config}")
    
    # Train model
    if config.model.model_type == 'pix2pix':
        train_pix2pix(models[0], models[1], train_loader, val_loader, config, device)
    elif config.model.model_type == 'cyclegan':
        train_cyclegan(models[0], models[1], models[2], models[3], train_loader, val_loader, config, device)
    elif config.model.model_type == 'edge_detection':
        print("Edge detection model doesn't require training")
        # Save the model
        save_checkpoint(models[0], None, 0, 0.0, f"{config.checkpoint_dir}/edge_detection.pth")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
