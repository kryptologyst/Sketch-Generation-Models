"""
Core neural network architectures for sketch generation.

This module contains implementations of:
- Pix2Pix Generator and Discriminator
- CycleGAN Generator and Discriminator  
- Edge Detection Baseline
- Utility functions for model initialization
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
        activation: str = 'leaky_relu'
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
            
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None
            
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransposeConvBlock(nn.Module):
    """Transpose convolutional block for upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
            
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None
            
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Pix2PixGenerator(nn.Module):
    """Pix2Pix Generator (U-Net architecture) for image-to-image translation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64, use_norm=False)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)
        self.enc6 = ConvBlock(512, 512)
        self.enc7 = ConvBlock(512, 512)
        self.enc8 = ConvBlock(512, 512, use_norm=False)
        
        # Decoder
        self.dec1 = TransposeConvBlock(512, 512, use_dropout=True)
        self.dec2 = TransposeConvBlock(1024, 512, use_dropout=True)
        self.dec3 = TransposeConvBlock(1024, 512, use_dropout=True)
        self.dec4 = TransposeConvBlock(1024, 512)
        self.dec5 = TransposeConvBlock(1024, 256)
        self.dec6 = TransposeConvBlock(512, 128)
        self.dec7 = TransposeConvBlock(256, 64)
        self.dec8 = TransposeConvBlock(128, out_channels, use_norm=False, activation='tanh')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        
        d8 = self.dec8(d7)
        
        return d8


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for Pix2Pix."""
    
    def __init__(self, in_channels: int = 4):  # 3 (image) + 1 (sketch)
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(in_channels, 64, use_norm=False),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512, stride=1, padding=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGANGenerator(nn.Module):
    """CycleGAN Generator (ResNet-based) for unpaired image translation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, n_residual_blocks: int = 9):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.down2 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(n_residual_blocks)
        ])
        
        # Upsampling
        self.up1 = TransposeConvBlock(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = TransposeConvBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class CycleGANDiscriminator(nn.Module):
    """CycleGAN Discriminator."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(in_channels, 64, use_norm=False),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512, stride=1, padding=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EdgeDetectionBaseline(nn.Module):
    """Simple edge detection baseline using Sobel operators."""
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Convert to 4D tensors for conv2d
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # Register as buffers so they move with the model
        self.register_buffer('sobel_x_buffer', self.sobel_x)
        self.register_buffer('sobel_y_buffer', self.sobel_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Sobel operators
        grad_x = F.conv2d(x, self.sobel_x_buffer, groups=3)
        grad_y = F.conv2d(x, self.sobel_y_buffer, groups=3)
        
        # Compute gradient magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Average across channels and normalize
        edge_magnitude = torch.mean(edge_magnitude, dim=1, keepdim=True)
        edge_magnitude = torch.tanh(edge_magnitude)  # Normalize to [-1, 1]
        
        return edge_magnitude


def init_weights(module: nn.Module) -> None:
    """Initialize network weights."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create a model instance based on type."""
    models = {
        'pix2pix_generator': Pix2PixGenerator,
        'pix2pix_discriminator': PatchGANDiscriminator,
        'cyclegan_generator': CycleGANGenerator,
        'cyclegan_discriminator': CycleGANDiscriminator,
        'edge_detection': EdgeDetectionBaseline
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = models[model_type](**kwargs)
    model.apply(init_weights)
    return model
