"""
Evaluation metrics for sketch generation models.

This module provides:
- FID (Fréchet Inception Distance) calculation
- LPIPS (Learned Perceptual Image Patch Similarity)
- Custom sketch-specific metrics
- Model comparison utilities
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchmetrics.image import FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity
import cv2
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


class SketchMetrics:
    """Collection of metrics for evaluating sketch generation quality."""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Initialize FID metric
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        
        # Initialize LPIPS metric
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        
        # Load pre-trained VGG for perceptual features
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        self.vgg.eval()
        
        # Edge detection kernel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(device)
    
    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Calculate FID between real and generated images."""
        # Normalize images to [0, 1] for FID
        real_norm = (real_images + 1) / 2
        fake_norm = (fake_images + 1) / 2
        
        # Clamp to valid range
        real_norm = torch.clamp(real_norm, 0, 1)
        fake_norm = torch.clamp(fake_norm, 0, 1)
        
        # Convert grayscale to RGB if needed
        if real_norm.size(1) == 1:
            real_norm = real_norm.repeat(1, 3, 1, 1)
        if fake_norm.size(1) == 1:
            fake_norm = fake_norm.repeat(1, 3, 1, 1)
        
        # Update FID metric
        self.fid.update(real_norm, real=True)
        self.fid.update(fake_norm, real=False)
        
        fid_score = self.fid.compute()
        self.fid.reset()
        
        return fid_score.item()
    
    def calculate_lpips(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Calculate LPIPS between real and generated images."""
        # Normalize images to [-1, 1] for LPIPS
        real_norm = torch.clamp(real_images, -1, 1)
        fake_norm = torch.clamp(fake_images, -1, 1)
        
        # Convert grayscale to RGB if needed
        if real_norm.size(1) == 1:
            real_norm = real_norm.repeat(1, 3, 1, 1)
        if fake_norm.size(1) == 1:
            fake_norm = fake_norm.repeat(1, 3, 1, 1)
        
        lpips_score = self.lpips(real_norm, fake_norm)
        return lpips_score.item()
    
    def calculate_edge_consistency(self, real_images: torch.Tensor, fake_sketches: torch.Tensor) -> float:
        """Calculate edge consistency between original images and generated sketches."""
        # Extract edges from original images
        real_edges = self._extract_edges(real_images)
        
        # Normalize fake sketches to [0, 1]
        fake_norm = (fake_sketches + 1) / 2
        fake_norm = torch.clamp(fake_norm, 0, 1)
        
        # Calculate edge consistency using L1 loss
        edge_loss = F.l1_loss(real_edges, fake_norm)
        
        return edge_loss.item()
    
    def calculate_sketch_quality(self, sketches: torch.Tensor) -> Dict[str, float]:
        """Calculate sketch-specific quality metrics."""
        # Normalize sketches to [0, 1]
        sketches_norm = (sketches + 1) / 2
        sketches_norm = torch.clamp(sketches_norm, 0, 1)
        
        # Calculate sparsity (percentage of white pixels)
        sparsity = torch.mean(sketches_norm).item()
        
        # Calculate edge density
        edges = self._extract_edges(sketches_norm)
        edge_density = torch.mean(edges).item()
        
        # Calculate contrast
        contrast = torch.std(sketches_norm).item()
        
        return {
            'sparsity': sparsity,
            'edge_density': edge_density,
            'contrast': contrast
        }
    
    def calculate_perceptual_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract perceptual features using VGG."""
        with torch.no_grad():
            # Normalize images to [0, 1] and convert to RGB
            images_norm = (images + 1) / 2
            images_norm = torch.clamp(images_norm, 0, 1)
            
            if images_norm.size(1) == 1:
                images_norm = images_norm.repeat(1, 3, 1, 1)
            
            # Extract features from VGG
            features = self.vgg(images_norm)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            
            return features
    
    def calculate_precision_recall(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> Dict[str, float]:
        """Calculate precision and recall for generated samples."""
        real_features = real_features.cpu().numpy()
        fake_features = fake_features.cpu().numpy()
        
        # Calculate pairwise distances
        real_distances = cdist(real_features, real_features)
        fake_distances = cdist(fake_features, fake_features)
        cross_distances = cdist(real_features, fake_features)
        
        # Calculate precision and recall
        precision = self._calculate_precision(real_distances, cross_distances)
        recall = self._calculate_recall(real_distances, cross_distances)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        }
    
    def _extract_edges(self, images: torch.Tensor) -> torch.Tensor:
        """Extract edges using Sobel operators."""
        if images.size(1) == 3:
            # Convert to grayscale
            gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        else:
            gray = images
        
        # Apply Sobel operators
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Calculate gradient magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to [0, 1]
        edge_magnitude = torch.tanh(edge_magnitude)
        
        return edge_magnitude
    
    def _calculate_precision(self, real_distances: np.ndarray, cross_distances: np.ndarray) -> float:
        """Calculate precision metric."""
        # For each fake sample, find closest real sample
        min_cross_distances = np.min(cross_distances, axis=0)
        
        # Calculate precision as fraction of fake samples that are close to real samples
        threshold = np.percentile(real_distances[real_distances > 0], 10)  # 10th percentile
        precision = np.mean(min_cross_distances < threshold)
        
        return precision
    
    def _calculate_recall(self, real_distances: np.ndarray, cross_distances: np.ndarray) -> float:
        """Calculate recall metric."""
        # For each real sample, find closest fake sample
        min_cross_distances = np.min(cross_distances, axis=1)
        
        # Calculate recall as fraction of real samples that have close fake samples
        threshold = np.percentile(real_distances[real_distances > 0], 10)  # 10th percentile
        recall = np.mean(min_cross_distances < threshold)
        
        return recall


class ModelEvaluator:
    """Comprehensive model evaluation utility."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = SketchMetrics(device)
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate a model on a dataset."""
        model.eval()
        
        all_real_images = []
        all_real_sketches = []
        all_fake_sketches = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_samples and i * dataloader.batch_size >= num_samples:
                    break
                
                real_images = batch['image'].to(self.device)
                real_sketches = batch['sketch'].to(self.device)
                
                # Generate fake sketches
                fake_sketches = model(real_images)
                
                all_real_images.append(real_images.cpu())
                all_real_sketches.append(real_sketches.cpu())
                all_fake_sketches.append(fake_sketches.cpu())
        
        # Concatenate all samples
        real_images = torch.cat(all_real_images, dim=0)
        real_sketches = torch.cat(all_real_sketches, dim=0)
        fake_sketches = torch.cat(all_fake_sketches, dim=0)
        
        if num_samples:
            real_images = real_images[:num_samples]
            real_sketches = real_sketches[:num_samples]
            fake_sketches = fake_sketches[:num_samples]
        
        # Calculate metrics
        results = {}
        
        # FID
        results['fid'] = self.metrics.calculate_fid(real_sketches, fake_sketches)
        
        # LPIPS
        results['lpips'] = self.metrics.calculate_lpips(real_sketches, fake_sketches)
        
        # Edge consistency
        results['edge_consistency'] = self.metrics.calculate_edge_consistency(real_images, fake_sketches)
        
        # Sketch quality
        sketch_quality = self.metrics.calculate_sketch_quality(fake_sketches)
        results.update(sketch_quality)
        
        # Perceptual features and precision/recall
        real_features = self.metrics.calculate_perceptual_features(real_sketches)
        fake_features = self.metrics.calculate_perceptual_features(fake_sketches)
        
        pr_metrics = self.metrics.calculate_precision_recall(real_features, fake_features)
        results.update(pr_metrics)
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models."""
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate_model(model, dataloader, num_samples)
        
        return results
    
    def create_evaluation_report(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> str:
        """Create a formatted evaluation report."""
        report = "Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Create table header
        metrics = list(next(iter(results.values())).keys())
        header = f"{'Model':<20}"
        for metric in metrics:
            header += f"{metric:<15}"
        report += header + "\n"
        report += "-" * len(header) + "\n"
        
        # Add results for each model
        for model_name, model_results in results.items():
            row = f"{model_name:<20}"
            for metric in metrics:
                value = model_results.get(metric, 0.0)
                row += f"{value:<15.4f}"
            report += row + "\n"
        
        # Add summary
        report += "\n" + "=" * 50 + "\n"
        report += "Summary:\n"
        
        for metric in metrics:
            values = [results[model][metric] for model in results.keys()]
            best_model = list(results.keys())[np.argmin(values)] if metric != 'precision' and metric != 'recall' and metric != 'f1' else list(results.keys())[np.argmax(values)]
            best_value = min(values) if metric != 'precision' and metric != 'recall' and metric != 'f1' else max(values)
            
            report += f"{metric}: {best_model} ({best_value:.4f})\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
