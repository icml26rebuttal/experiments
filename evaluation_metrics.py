"""
Simplified Evaluation Metrics for Adversarial Attacks
======================================================

Computes essential metrics for adversarial examples:
- Classification metrics: clean_accuracy, adversarial_accuracy, attack_success_rate
- Perturbation metrics: L2, L∞, Gabor frame norm (mean + std)
- Perceptual metrics: LPIPS, SSIM (mean + std)

Usage:
    from evaluation_metrics import AdversarialMetrics
    
    evaluator = AdversarialMetrics(device='cuda', M=M, D=D)
    metrics = evaluator.compute_all_metrics(model, x_clean, x_adv, y_true)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

# Optional imports with graceful degradation
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


class AdversarialMetrics:
    """
    Simplified adversarial attack evaluation metrics
    
    Essential Metrics:
    - clean_accuracy: Accuracy on clean images
    - adversarial_accuracy: Accuracy on adversarial images
    - attack_success_rate: Percentage of successful attacks
    - mean_l2_norm, std_l2_norm: L2 perturbation norm
    - mean_linf_norm, std_linf_norm: L∞ perturbation norm
    - mean_gabor_frame_norm, std_gabor_frame_norm: Gabor frame norm
    - lpips_mean, lpips_std: LPIPS perceptual distance
    - ssim_mean, ssim_std: SSIM structural similarity
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        lpips_net: str = 'alex',
        verbose: bool = True,
        M: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        case_type: int = 1
    ):
        """
        Initialize metrics evaluator
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            lpips_net: LPIPS network ('alex', 'vgg', or 'squeeze')
            verbose: Print warnings for missing libraries
            M: Gabor matrix M = Psi^H * D * Psi for Case 1 metric
            D: Diagonal matrix for Gabor frame (for Case 2)
            tau: Soft-thresholding parameter (for Case 1)
            case_type: 1 for Case 1 metric, 2 for Case 2 metric
        """
        self.device = device
        self.verbose = verbose
        
        # Store Gabor operators
        self.M = M
        self.tau = tau if tau is not None else 0.1
        self.case_type = case_type
        
        # Initialize LPIPS if available
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(device)
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        else:
            self.lpips_model = None
            if verbose:
                print("Warning: LPIPS not available. Install with: pip install lpips")
        
        if not SSIM_AVAILABLE and verbose:
            print("Warning: SSIM not available. Install with: pip install pytorch-msssim")
    
    def compute_classification_metrics(
        self,
        model: nn.Module,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute classification accuracy metrics
        
        Returns:
            clean_accuracy: Accuracy on clean images
            adversarial_accuracy: Accuracy on adversarial images
            attack_success_rate: # successful adversarial examples / # total adversarial examples
                                 where successful = model misclassifies the adversarial example
        """
        model.eval()
        with torch.no_grad():
            # Clean predictions
            outputs_clean = model(x_clean)
            pred_clean = outputs_clean.argmax(dim=1)
            clean_correct = (pred_clean == y_true)
            
            # Adversarial predictions
            outputs_adv = model(x_adv)
            pred_adv = outputs_adv.argmax(dim=1)
            adv_correct = (pred_adv == y_true)
            
            attack_success = ~adv_correct  # Adversarial examples that fooled the model
            
            metrics = {
                'clean_accuracy': clean_correct.float().mean().item(),
                'adversarial_accuracy': adv_correct.float().mean().item(),
                'attack_success_rate': attack_success.float().mean().item()
            }
        
        return metrics
    
    def compute_perturbation_metrics_gabor(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        eps_dgf,
        last_delta
    ) -> Dict[str, float]:
        """
        Compute perturbation norms: L2, L∞, and Gabor frame norm
        
        Returns:
            mean_l2_norm, std_l2_norm
            mean_linf_norm, std_linf_norm
            mean_gabor_frame_norm, std_gabor_frame_norm
        """
        perturbation = x_adv - x_clean
        B = perturbation.shape[0]
        
        # L2 norm per sample: ||δ||_2
        l2_norms = torch.norm(perturbation.view(B, -1), p=2, dim=1)
        
        # L∞ norm per sample: ||δ||_∞
        linf_norms = torch.norm(perturbation.view(B, -1), p=float('inf'), dim=1)
        
        metrics = {
            'mean_l2_norm': l2_norms.mean().item(),
            'std_l2_norm': l2_norms.std().item(),
            'mean_linf_norm': linf_norms.mean().item(),
            'std_linf_norm': linf_norms.std().item()
        }
        
        # Gabor frame norm (case-specific)
        gabor_metrics = self._compute_gabor_norm(x_clean, x_adv, eps_dgf, last_delta)
        metrics.update(gabor_metrics)
        
        return metrics
    
    def _compute_gabor_norm(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        eps_dgf,
        last_delta
    ) -> Dict[str, float]:
        """
        Compute Gabor frame norm (case-specific)
        
        Case 1: ||w||_D = √(δ^T M δ) where δ = x_adv - Ψ† S_τ(Ψ x_clean)
        """
        q, feasible = self._compute_case1_norm(x_clean, x_adv, self.M, eps_dgf, last_delta)
        
        return {
            'mean_gabor_frame_norm': q.mean().item(),
            "feasible_frac": feasible.float().mean().item()
        }
    
    def _compute_case1_norm(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        M: torch.Tensor,
        eps_dgf: float,
        last_delta
    ) -> list:
        """Compute Case 1 Gabor norm"""

        device = x_clean.device
        delta_c = last_delta.to(M.dtype)
        
        q = torch.einsum("kn,nm,km->k", delta_c.conj(), M.to(device), delta_c).real  # (K,)

        eps2 = eps_dgf * eps_dgf
        feasible = (q <= eps2 * (1 + 1e-6))

        return q, feasible
    
    def compute_perturbation_metrics(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute perturbation norms: L2, L∞, and Gabor frame norm
        
        Returns:
            mean_l2_norm, std_l2_norm
            mean_linf_norm, std_linf_norm
            mean_gabor_frame_norm, std_gabor_frame_norm
        """
        perturbation = x_adv - x_clean
        B = perturbation.shape[0]
        
        # L2 norm per sample: ||δ||_2
        l2_norms = torch.norm(perturbation.view(B, -1), p=2, dim=1)
        
        # L∞ norm per sample: ||δ||_∞
        linf_norms = torch.norm(perturbation.view(B, -1), p=float('inf'), dim=1)
        
        metrics = {
            'mean_l2_norm': l2_norms.mean().item(),
            'std_l2_norm': l2_norms.std().item(),
            'mean_linf_norm': linf_norms.mean().item(),
            'std_linf_norm': linf_norms.std().item()
        }
        
        return metrics
    

    def compute_perceptual_metrics(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute perceptual metrics: LPIPS and SSIM
        
        Returns:
            lpips_mean, lpips_std
            ssim_mean, ssim_std
        """
        metrics = {}
        
        # LPIPS
        if self.lpips_model is not None:
            with torch.no_grad():
                # LPIPS expects images in [-1, 1]
                x_clean_normalized = x_clean * 2 - 1
                x_adv_normalized = x_adv * 2 - 1
                
                lpips_values = self.lpips_model(x_clean_normalized, x_adv_normalized)
                lpips_values = lpips_values.view(-1)
                
                metrics['lpips_mean'] = lpips_values.mean().item()
                metrics['lpips_std'] = lpips_values.std().item()
        else:
            metrics['lpips_mean'] = None
            metrics['lpips_std'] = None
        
        # SSIM
        if SSIM_AVAILABLE:
            with torch.no_grad():
                # SSIM per sample
                B = x_clean.shape[0]
                ssim_values = []
                for i in range(B):
                    ssim_val = ssim(
                        x_clean[i:i+1], x_adv[i:i+1],
                        data_range=1.0, size_average=True
                    )
                    ssim_values.append(ssim_val.item())
                
                metrics['ssim_mean'] = np.mean(ssim_values)
                metrics['ssim_std'] = np.std(ssim_values)
        else:
            metrics['ssim_mean'] = None
            metrics['ssim_std'] = None
        
        return metrics
    
    def compute_all_metrics_gabor(
        self,
        model: nn.Module,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        y_true: torch.Tensor,
        eps_dgf,
        last_delta
    ) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns dictionary with:
        - clean_accuracy
        - adversarial_accuracy
        - attack_success_rate
        - mean_l2_norm, std_l2_norm
        - mean_linf_norm, std_linf_norm
        - mean_gabor_frame_norm, std_gabor_frame_norm
        - lpips_mean, lpips_std
        - ssim_mean, ssim_std
        """
        metrics = {}
        
        # Classification metrics
        metrics.update(self.compute_classification_metrics(model, x_clean, x_adv, y_true))
        
        # Perturbation metrics
        metrics.update(self.compute_perturbation_metrics_gabor(x_clean, x_adv, eps_dgf, last_delta))

        # Perceptual metrics
        metrics.update(self.compute_perceptual_metrics(x_clean, x_adv))
        
        return metrics
    
    def compute_all_metrics(
        self,
        model: nn.Module,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns dictionary with:
        - clean_accuracy
        - adversarial_accuracy
        - attack_success_rate
        - mean_l2_norm, std_l2_norm
        - mean_linf_norm, std_linf_norm
        - mean_gabor_frame_norm, std_gabor_frame_norm
        - lpips_mean, lpips_std
        - ssim_mean, ssim_std
        """
        metrics = {}
        
        # Classification metrics
        metrics.update(self.compute_classification_metrics(model, x_clean, x_adv, y_true))
        
        # Perturbation metrics
        metrics.update(self.compute_perturbation_metrics(x_clean, x_adv))

        # Perceptual metrics
        metrics.update(self.compute_perceptual_metrics(x_clean, x_adv))
        
        return metrics