"""
Unit tests for sketch generation models.

This module contains tests for:
- Model architectures
- Data loading
- Training utilities
- Evaluation metrics
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model, Pix2PixGenerator, CycleGANGenerator, EdgeDetectionBaseline
from src.data import PairedImageSketchDataset, SyntheticSketchDataset, get_transforms
from src.utils import GANLoss, L1Loss, CycleConsistencyLoss
from src.utils.metrics import SketchMetrics


class TestModels:
    """Test model architectures."""
    
    def test_pix2pix_generator(self):
        """Test Pix2Pix generator."""
        model = create_model('pix2pix_generator', in_channels=3, out_channels=1)
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (1, 1, 256, 256)
        assert torch.all(output >= -1) and torch.all(output <= 1)
    
    def test_cyclegan_generator(self):
        """Test CycleGAN generator."""
        model = create_model('cyclegan_generator', in_channels=3, out_channels=1)
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (1, 1, 256, 256)
        assert torch.all(output >= -1) and torch.all(output <= 1)
    
    def test_edge_detection(self):
        """Test edge detection baseline."""
        model = create_model('edge_detection')
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (1, 1, 256, 256)
        assert torch.all(output >= -1) and torch.all(output <= 1)
    
    def test_discriminators(self):
        """Test discriminator models."""
        # Test Pix2Pix discriminator
        pix2pix_disc = create_model('pix2pix_discriminator', in_channels=4)
        x = torch.randn(1, 4, 256, 256)
        output = pix2pix_disc(x)
        assert output.shape[0] == 1  # Batch size
        
        # Test CycleGAN discriminator
        cyclegan_disc = create_model('cyclegan_discriminator', in_channels=1)
        x = torch.randn(1, 1, 256, 256)
        output = cyclegan_disc(x)
        assert output.shape[0] == 1  # Batch size


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticSketchDataset(num_samples=10, image_size=(64, 64))
        
        assert len(dataset) == 10
        
        # Test data loading
        sample = dataset[0]
        assert 'image' in sample
        assert 'sketch' in sample
        assert sample['image'].shape == (3, 64, 64)
        assert sample['sketch'].shape == (1, 64, 64)
    
    def test_transforms(self):
        """Test data transforms."""
        transform = get_transforms(image_size=(64, 64), mode='train', use_augmentation=False)
        
        # Test transform
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        transformed = transform(image=image)
        
        assert 'image' in transformed
        assert transformed['image'].shape == (3, 64, 64)


class TestLossFunctions:
    """Test loss functions."""
    
    def test_gan_loss(self):
        """Test GAN loss functions."""
        # Test LSGAN loss
        lsgan_loss = GANLoss(mode='lsgan')
        pred = torch.randn(1, 1, 16, 16)
        
        real_loss = lsgan_loss(pred, True)
        fake_loss = lsgan_loss(pred, False)
        
        assert real_loss.item() > 0
        assert fake_loss.item() > 0
    
    def test_l1_loss(self):
        """Test L1 loss."""
        l1_loss = L1Loss(weight=10.0)
        
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randn(1, 1, 64, 64)
        
        loss = l1_loss(pred, target)
        assert loss.item() > 0
    
    def test_cycle_loss(self):
        """Test cycle consistency loss."""
        cycle_loss = CycleConsistencyLoss(weight=10.0)
        
        real = torch.randn(1, 3, 64, 64)
        reconstructed = torch.randn(1, 3, 64, 64)
        
        loss = cycle_loss(real, reconstructed)
        assert loss.item() > 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_sketch_metrics(self):
        """Test sketch metrics calculation."""
        device = torch.device('cpu')
        metrics = SketchMetrics(device)
        
        # Create test data
        real_images = torch.randn(2, 3, 64, 64)
        fake_images = torch.randn(2, 1, 64, 64)
        
        # Test sketch quality metrics
        quality_metrics = metrics.calculate_sketch_quality(fake_images)
        
        assert 'sparsity' in quality_metrics
        assert 'edge_density' in quality_metrics
        assert 'contrast' in quality_metrics
        
        assert 0 <= quality_metrics['sparsity'] <= 1
        assert 0 <= quality_metrics['edge_density'] <= 1
        assert quality_metrics['contrast'] >= 0


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self):
        """Test a single training step."""
        device = torch.device('cpu')
        
        # Create models
        generator = create_model('pix2pix_generator')
        discriminator = create_model('pix2pix_discriminator')
        
        # Create sample batch
        batch = {
            'image': torch.randn(2, 3, 64, 64),
            'sketch': torch.randn(2, 1, 64, 64)
        }
        
        # Test forward passes
        fake_sketch = generator(batch['image'])
        assert fake_sketch.shape == batch['sketch'].shape
        
        real_pairs = torch.cat([batch['image'], batch['sketch']], dim=1)
        fake_pairs = torch.cat([batch['image'], fake_sketch], dim=1)
        
        d_real = discriminator(real_pairs)
        d_fake = discriminator(fake_pairs)
        
        assert d_real.shape[0] == batch['image'].shape[0]
        assert d_fake.shape[0] == batch['image'].shape[0]


if __name__ == '__main__':
    pytest.main([__file__])
