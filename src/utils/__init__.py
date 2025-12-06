"""
Training utilities, loss functions, and optimization for sketch generation models.

This module provides:
- Loss functions for GAN training (adversarial, cycle, identity, etc.)
- Training loops for different model types
- Optimization utilities
- Checkpointing and logging
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class GANLoss(nn.Module):
    """GAN loss with different modes."""
    
    def __init__(self, mode: str = 'lsgan', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.mode = mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'nsgan':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'wgan':
            self.loss = None  # WGAN uses raw output
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.mode == 'lsgan':
            if target_is_real:
                target = torch.ones_like(prediction) * self.target_real_label
            else:
                target = torch.zeros_like(prediction) * self.target_fake_label
            return self.loss(prediction, target)
        
        elif self.mode == 'nsgan':
            if target_is_real:
                target = torch.ones_like(prediction)
            else:
                target = torch.zeros_like(prediction)
            return self.loss(prediction, target)
        
        elif self.mode == 'wgan':
            if target_is_real:
                return -torch.mean(prediction)
            else:
                return torch.mean(prediction)


class L1Loss(nn.Module):
    """L1 loss for reconstruction."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.loss(pred, target)


class CycleConsistencyLoss(nn.Module):
    """Cycle consistency loss for CycleGAN."""
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        self.loss = nn.L1Loss()
    
    def forward(self, real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        return self.weight * self.loss(real, reconstructed)


class IdentityLoss(nn.Module):
    """Identity loss for CycleGAN."""
    
    def __init__(self, weight: float = 5.0):
        super().__init__()
        self.weight = weight
        self.loss = nn.L1Loss()
    
    def forward(self, real: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.weight * self.loss(real, identity)


class GradientPenalty(nn.Module):
    """Gradient penalty for WGAN-GP."""
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
        
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        d_interpolated = discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        penalty = self.weight * torch.mean((gradient_norm - 1) ** 2)
        
        return penalty


class Pix2PixTrainer:
    """Trainer for Pix2Pix model."""
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        lr: float = 0.0002,
        beta1: float = 0.5,
        lambda_l1: float = 100.0,
        gan_mode: str = 'lsgan'
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Loss functions
        self.gan_loss = GANLoss(mode=gan_mode)
        self.l1_loss = L1Loss(weight=lambda_l1)
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.LinearLR(self.optimizer_g, start_factor=1.0, end_factor=0.0, total_iters=100)
        self.scheduler_d = optim.lr_scheduler.LinearLR(self.optimizer_d, start_factor=1.0, end_factor=0.0, total_iters=100)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        real_images = batch['image'].to(self.device)
        real_sketches = batch['sketch'].to(self.device)
        
        batch_size = real_images.size(0)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        # Real pairs
        real_pairs = torch.cat([real_images, real_sketches], dim=1)
        d_real = self.discriminator(real_pairs)
        d_real_loss = self.gan_loss(d_real, True)
        
        # Fake pairs
        fake_sketches = self.generator(real_images)
        fake_pairs = torch.cat([real_images, fake_sketches.detach()], dim=1)
        d_fake = self.discriminator(fake_pairs)
        d_fake_loss = self.gan_loss(d_fake, False)
        
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        fake_sketches = self.generator(real_images)
        fake_pairs = torch.cat([real_images, fake_sketches], dim=1)
        d_fake = self.discriminator(fake_pairs)
        
        g_gan_loss = self.gan_loss(d_fake, True)
        g_l1_loss = self.l1_loss(fake_sketches, real_sketches)
        g_loss = g_gan_loss + g_l1_loss
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_gan_loss': g_gan_loss.item(),
            'g_l1_loss': g_l1_loss.item()
        }
    
    def generate_samples(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate samples for visualization."""
        with torch.no_grad():
            real_images = batch['image'].to(self.device)
            fake_sketches = self.generator(real_images)
            return fake_sketches


class CycleGANTrainer:
    """Trainer for CycleGAN model."""
    
    def __init__(
        self,
        g_image_to_sketch: nn.Module,
        g_sketch_to_image: nn.Module,
        d_image: nn.Module,
        d_sketch: nn.Module,
        device: torch.device,
        lr: float = 0.0002,
        beta1: float = 0.5,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        gan_mode: str = 'lsgan'
    ):
        self.g_image_to_sketch = g_image_to_sketch.to(device)
        self.g_sketch_to_image = g_sketch_to_image.to(device)
        self.d_image = d_image.to(device)
        self.d_sketch = d_sketch.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            list(g_image_to_sketch.parameters()) + list(g_sketch_to_image.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_d = optim.Adam(
            list(d_image.parameters()) + list(d_sketch.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )
        
        # Loss functions
        self.gan_loss = GANLoss(mode=gan_mode)
        self.cycle_loss = CycleConsistencyLoss(weight=lambda_cycle)
        self.identity_loss = IdentityLoss(weight=lambda_identity)
    
    def train_step(self, batch_image: Dict[str, torch.Tensor], batch_sketch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        real_images = batch_image['image'].to(self.device)
        real_sketches = batch_sketch['sketch'].to(self.device)
        
        # Train Generators
        self.optimizer_g.zero_grad()
        
        # Identity loss
        id_images = self.g_sketch_to_image(real_images)
        id_sketches = self.g_image_to_sketch(real_sketches)
        
        id_loss_images = self.identity_loss(real_images, id_images)
        id_loss_sketches = self.identity_loss(real_sketches, id_sketches)
        
        # GAN loss
        fake_sketches = self.g_image_to_sketch(real_images)
        fake_images = self.g_sketch_to_image(real_sketches)
        
        d_fake_sketches = self.d_sketch(fake_sketches)
        d_fake_images = self.d_image(fake_images)
        
        g_loss_sketches = self.gan_loss(d_fake_sketches, True)
        g_loss_images = self.gan_loss(d_fake_images, True)
        
        # Cycle loss
        rec_images = self.g_sketch_to_image(fake_sketches)
        rec_sketches = self.g_image_to_sketch(fake_images)
        
        cycle_loss_images = self.cycle_loss(real_images, rec_images)
        cycle_loss_sketches = self.cycle_loss(real_sketches, rec_sketches)
        
        g_loss = (
            g_loss_sketches + g_loss_images +
            cycle_loss_images + cycle_loss_sketches +
            id_loss_images + id_loss_sketches
        )
        
        g_loss.backward()
        self.optimizer_g.step()
        
        # Train Discriminators
        self.optimizer_d.zero_grad()
        
        # Image discriminator
        d_real_images = self.d_image(real_images)
        d_fake_images = self.d_image(fake_images.detach())
        
        d_loss_images = self.gan_loss(d_real_images, True) + self.gan_loss(d_fake_images, False)
        
        # Sketch discriminator
        d_real_sketches = self.d_sketch(real_sketches)
        d_fake_sketches = self.d_sketch(fake_sketches.detach())
        
        d_loss_sketches = self.gan_loss(d_real_sketches, True) + self.gan_loss(d_fake_sketches, False)
        
        d_loss = (d_loss_images + d_loss_sketches) * 0.5
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'cycle_loss_images': cycle_loss_images.item(),
            'cycle_loss_sketches': cycle_loss_sketches.item(),
            'id_loss_images': id_loss_images.item(),
            'id_loss_sketches': id_loss_sketches.item()
        }
    
    def generate_samples(self, batch_image: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate samples for visualization."""
        with torch.no_grad():
            real_images = batch_image['image'].to(self.device)
            fake_sketches = self.g_image_to_sketch(real_images)
            rec_images = self.g_sketch_to_image(fake_sketches)
            
            return {
                'fake_sketches': fake_sketches,
                'rec_images': rec_images
            }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    filepath: str
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_sample_grid(
    images: torch.Tensor,
    sketches: torch.Tensor,
    generated: torch.Tensor,
    num_samples: int = 8
) -> plt.Figure:
    """Create a grid visualization of samples."""
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    for i in range(num_samples):
        # Original images
        img = images[i].cpu().permute(1, 2, 0)
        img = (img + 1) / 2  # Denormalize
        img = torch.clamp(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Real sketches
        sketch = sketches[i].cpu().squeeze()
        sketch = (sketch + 1) / 2  # Denormalize
        sketch = torch.clamp(sketch, 0, 1)
        axes[1, i].imshow(sketch, cmap='gray')
        axes[1, i].set_title('Real Sketch')
        axes[1, i].axis('off')
        
        # Generated sketches
        gen = generated[i].cpu().squeeze()
        gen = (gen + 1) / 2  # Denormalize
        gen = torch.clamp(gen, 0, 1)
        axes[2, i].imshow(gen, cmap='gray')
        axes[2, i].set_title('Generated')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig
