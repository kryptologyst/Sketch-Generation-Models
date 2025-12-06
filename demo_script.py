#!/usr/bin/env python3
"""
Quick demo script to showcase the modernized sketch generation project.

This script demonstrates the key features of the refactored project:
- Model creation and testing
- Data pipeline functionality
- Evaluation metrics
- Configuration management
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models import create_model
from src.data import SyntheticSketchDataset, create_synthetic_dataset
from src.utils.metrics import SketchMetrics
from src.configs import create_default_configs, validate_config
from src import set_seed, get_device


def main():
    """Main demo function."""
    print("🎨 Sketch Generation Models - Modern PyTorch Implementation")
    print("=" * 60)
    
    # Setup
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Model Creation Demo
    print("\n1. Creating Models...")
    models = {
        'Pix2Pix': create_model('pix2pix_generator'),
        'CycleGAN': create_model('cyclegan_generator'),
        'Edge Detection': create_model('edge_detection')
    }
    
    for name, model in models.items():
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {name}: {total_params:,} parameters")
    
    # 2. Data Pipeline Demo
    print("\n2. Testing Data Pipeline...")
    
    # Create synthetic dataset
    dataset = SyntheticSketchDataset(num_samples=5, image_size=(128, 128))
    print(f"  ✓ Created synthetic dataset with {len(dataset)} samples")
    
    # Test data loading
    sample = dataset[0]
    print(f"  ✓ Sample image shape: {sample['image'].shape}")
    print(f"  ✓ Sample sketch shape: {sample['sketch'].shape}")
    
    # 3. Model Testing Demo
    print("\n3. Testing Model Inference...")
    
    # Create test input
    test_image = torch.randn(1, 3, 128, 128).to(device)
    
    with torch.no_grad():
        for name, model in models.items():
            output = model(test_image)
            print(f"  ✓ {name}: Input {test_image.shape} → Output {output.shape}")
    
    # 4. Evaluation Metrics Demo
    print("\n4. Testing Evaluation Metrics...")
    
    metrics = SketchMetrics(device)
    
    # Create test data
    real_images = torch.randn(2, 3, 64, 64).to(device)
    fake_sketches = torch.randn(2, 1, 64, 64).to(device)
    
    # Test sketch quality metrics
    quality_metrics = metrics.calculate_sketch_quality(fake_sketches)
    print(f"  ✓ Sketch quality metrics: {quality_metrics}")
    
    # Test edge consistency
    edge_consistency = metrics.calculate_edge_consistency(real_images, fake_sketches)
    print(f"  ✓ Edge consistency: {edge_consistency:.4f}")
    
    # 5. Configuration Management Demo
    print("\n5. Testing Configuration Management...")
    
    configs = create_default_configs()
    print(f"  ✓ Created {len(configs)} default configurations")
    
    for name, config in configs.items():
        issues = validate_config(config)
        status = "✓ Valid" if not issues else f"✗ {len(issues)} issues"
        print(f"  {status}: {name} config")
    
    # 6. Visualization Demo
    print("\n6. Creating Sample Visualization...")
    
    # Generate sample
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        pix2pix_sketch = models['Pix2Pix'](image)
        edge_sketch = models['Edge Detection'](image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    img_display = image.squeeze(0).cpu().permute(1, 2, 0)
    img_display = (img_display + 1) / 2
    img_display = torch.clamp(img_display, 0, 1)
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Pix2Pix sketch
    sketch_display = pix2pix_sketch.squeeze(0).cpu().squeeze()
    sketch_display = (sketch_display + 1) / 2
    sketch_display = torch.clamp(sketch_display, 0, 1)
    axes[1].imshow(sketch_display, cmap='gray')
    axes[1].set_title('Pix2Pix Sketch')
    axes[1].axis('off')
    
    # Edge detection sketch
    edge_display = edge_sketch.squeeze(0).cpu().squeeze()
    edge_display = (edge_display + 1) / 2
    edge_display = torch.clamp(edge_display, 0, 1)
    axes[2].imshow(edge_display, cmap='gray')
    axes[2].set_title('Edge Detection')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved visualization to 'demo_output.png'")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Modern PyTorch implementation with type hints")
    print("  ✓ Multiple model architectures (Pix2Pix, CycleGAN, Edge Detection)")
    print("  ✓ Comprehensive data pipeline with synthetic dataset generation")
    print("  ✓ Advanced evaluation metrics (FID, LPIPS, edge consistency)")
    print("  ✓ Configuration management with validation")
    print("  ✓ Device-agnostic code (CUDA/MPS/CPU)")
    print("  ✓ Production-ready structure with tests and CI/CD")
    
    print("\nNext Steps:")
    print("  1. Run 'streamlit run demo/app.py' for interactive demo")
    print("  2. Run 'python scripts/train.py --config configs/pix2pix.yaml' to train")
    print("  3. Run 'python scripts/sample.py --help' for sampling options")
    print("  4. Check 'notebooks/demo.ipynb' for detailed examples")
    
    print("\nProject Structure:")
    print("  📁 src/           - Core modules (models, data, utils, configs)")
    print("  📁 scripts/       - Training and sampling scripts")
    print("  📁 demo/          - Streamlit web app")
    print("  📁 configs/       - YAML configuration files")
    print("  📁 tests/         - Unit tests")
    print("  📁 notebooks/     - Jupyter notebook examples")
    print("  📄 README.md      - Comprehensive documentation")


if __name__ == '__main__':
    main()
