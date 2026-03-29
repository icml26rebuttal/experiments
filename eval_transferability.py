"""
CIFAR10 Adversarial Attack Transferability Evaluation
======================================================

Evaluates the transferability of adversarial examples across different models.
For each source model, generates adversarial examples and tests them on all target models.

Transferability is measured by:
- Transfer Attack Success Rate (Transfer ASR): % of adversarial examples from source 
  that fool the target model
- Transfer Accuracy: Accuracy of target model on adversarial examples from source
- Cross-model comparison matrices

Usage:
    # Evaluate transferability for Case 1
    python eval_transferability_cifar10.py --case case1
    
    # Evaluate with specific models
    python eval_transferability_cifar10.py --case case1 --models Ding2020MMA Rice2020Overfitting
    
    # Quick test with fewer samples
    python eval_transferability_cifar10.py --case case1 --num-samples 500
"""
import math
import gc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import attack and metrics classes
from transforms import *
from dgf_pgd import DGFPGDAttack
from evaluation_metrics import AdversarialMetrics


# ============================================================================
# Command-line Arguments
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CIFAR10 Adversarial Attack Transferability Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-source',
        type=str,
        default='robustbench',
        choices=['robustbench', 'pretrained'],
        help='Model source: robustbench or pretrained models'
    )
    
    # Case selection
    parser.add_argument(
        "--case",
        default="case1",
        type=str,
        choices=["case1", "case2", "case3", "case4"],
        help="Attack case to evaluate (case4 = AutoAttack)",
    )

    # AutoAttack parameters (case4)
    parser.add_argument(
        "--aa-norm",
        type=str,
        default="Linf",
        choices=["Linf", "L2"],
        help="AutoAttack threat model norm (case4 only)",
    )
    parser.add_argument(
        "--aa-version",
        type=str,
        default="standard",
        choices=["standard", "plus", "rand"],
        help="AutoAttack version (case4 only)",
    )
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='./data/cifar10', 
                       help='CIFAR10 data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=1000, 
                       help='Number of test samples')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Data loading workers')
    
    # Attack parameters
    parser.add_argument('--epsilon', type=float, default=8/255,
                       help='Attack epsilon (of spatial ball)')
    parser.add_argument('--gamma', type=float, default=0.1, help='Step size')
    parser.add_argument('--num-steps', type=int, default=20, help='PGD iterations')
    parser.add_argument('--tau', type=float, default=0.5,
                       help='Soft-thresholding parameter (Case 1)')
    parser.add_argument('--a', type=int, default=1, help='Time lattice parameter')
    parser.add_argument('--b', type=int, default=16, help='Frequency lattice parameter')
    parser.add_argument('--rho', type=float, default=1.0, help='Frame potential factor')
    parser.add_argument('--window-type', type=str, default='Hann',
                       choices=['Hann', 'Blackman', 'Gaussian'],
                       help='Type of Gabor window function')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to evaluate (default: all available)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./results/transferability', 
                       help='Output directory')
    parser.add_argument('--lpips-net', type=str, default='alex', 
                       choices=['alex', 'vgg', 'squeeze'])
    parser.add_argument('--save-heatmaps', action='store_true', 
                       help='Save transferability heatmaps')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


# ============================================================================
# Dataset Loading
# ============================================================================

def load_cifar10(args):
    """Load CIFAR10 test data"""
    print("\nLoading CIFAR10 dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    # Subset if requested
    if args.num_samples is not None and args.num_samples < len(testset):
        indices = np.random.choice(len(testset), args.num_samples, replace=False)
        testset = Subset(testset, indices)
    
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Loaded {len(testset)} test samples")
    return testloader


# ============================================================================
# Model Loading
# ============================================================================

def load_cifar10_models(args):
    """Load pretrained CIFAR10 models"""
        
    if args.model_source == 'robustbench':
        return load_robustbench_models(args)
    elif args.model_source == 'pretrained':
        return load_pretrained_backbones(args)
    else:
        raise ValueError(f"Unknown model source: {args.model_source}")


def load_robustbench_models(args):
    """Load pretrained CIFAR10 models from RobustBench"""
    print("\nLoading CIFAR10 models from RobustBench...")
    
    try:
        from robustbench.utils import load_model as rb_load_model
    except ImportError:
        print("\nERROR: robustbench not installed!")
        print("Install with: pip install robustbench")
        raise ImportError("robustbench is required for RobustBench models")
    
    # Available RobustBench CIFAR10 models
    model_names = [
        'Amini2024MeanSparse_Ra_WRN_70_16',
        'Bai2024MixedNUTS',
        'Ding2020MMA',
        'Engstrom2019Robustness',
        'Rade2021Helper_R18_ddpm',
        'Augustin2020Adversarial',
        'Rebuffi2021Fixing_28_10_cutmix_ddpm',
        'Rice2020Overfitting',
        'Rony2019Decoupling',
        'Debenedetti2022Light_XCiT-L12',
        'Debenedetti2022Light_XCiT-M12',
        'Debenedetti2022Light_XCiT-S12',
        'Wu2020Adversarial',
    ]
    
    # Filter if specific models requested
    if args.models:
        model_names = [m for m in model_names if m in args.models]
    
    if not model_names:
        raise ValueError(f"No valid models specified. Available: {model_names}")
    
    print(f"Loading {len(model_names)} models:")
    for name in model_names:
        print(f"  - {name}")
    
    models_dict = {}
    
    for model_name in model_names:
        try:
            print(f"\n  Loading {model_name}...")
            model = rb_load_model(
                model_name=model_name,
                dataset="cifar10",
                threat_model="Linf"
            ).to(args.device)
            models_dict[model_name] = model.eval()
            print(f"    ✓ Success")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:100]}")
    
    if not models_dict:
        raise RuntimeError("No models loaded successfully!")
    
    print(f"\n✓ Successfully loaded {len(models_dict)} models")
    return models_dict


def load_pretrained_backbones(args):
    """Load pretrained backbone models from chenyaofo/pytorch-cifar-models"""
    print("\nLoading pretrained backbone models from PyTorch Hub...")
    
    # Available pretrained backbones
    available_backbones = {
        'resnet44': 'cifar10_resnet44',
        'resnet56': 'cifar10_resnet56',
        'vgg13_bn': 'cifar10_vgg13_bn',
        'vgg16_bn': 'cifar10_vgg16_bn',
        'mobilenetv2_x0_75': 'cifar10_mobilenetv2_x0_75',
        'mobilenetv2_x1_0': 'cifar10_mobilenetv2_x1_0',
        'repvgg_a1': 'cifar10_repvgg_a1',
        'repvgg_a2': 'cifar10_repvgg_a2',
    }
    
    # Filter if specific models requested
    if args.models:
        backbones_to_load = {k: v for k, v in available_backbones.items() 
                            if k in args.models}
    else:
        backbones_to_load = available_backbones
    
    if not backbones_to_load:
        raise ValueError(f"No valid models. Available: {list(available_backbones.keys())}")
    
    print(f"Loading {len(backbones_to_load)} backbone models:")
    for name in backbones_to_load.keys():
        print(f"  - {name}")
    
    models_dict = {}
    
    for model_name, hub_name in backbones_to_load.items():
        try:
            print(f"\n  Loading {model_name}...")
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                hub_name,
                pretrained=True
            ).to(args.device)
            models_dict[model_name] = model.eval()
            print(f"    ✓ Success")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:100]}")
    
    if not models_dict:
        raise RuntimeError("No models loaded successfully!")
    
    print(f"\n✓ Successfully loaded {len(models_dict)} models")
    return models_dict

def generate_gabor_operators_cifar10(device, a, b, rho, window_type='Hann'):
    """
    Generate Gabor frame operators for CIFAR10 (32X32 per channel)
    
    Returns:
        Tuple of operators
    """

    print("\nGenerating 2D Gabor operators (and their kronecker products for CIFAR10...")
    
    # Parameters
    H, W = 32, 32
    n = H * W  # 1024
    
    print(f"  Dimensions: H={H}, W={W}, N={(n)/(a*b)}")
    
    print("  Generating Ψ...")
    Psi_2D = DGT(H, a=a, b=b, window=window_type)
    Psi_2D = Psi_2D/torch.linalg.norm(Psi_2D, dim = 1, keepdim=True)
    
    print("  Computing Ψ^+...")
    Psi_plus_2D = torch.linalg.pinv(Psi_2D)
    
    print("  Generating D...")
    D_2D = diag_weights_from_mc_row_sums(Psi_2D, mode='down')

    S_2D = frameop_DGT(Psi_2D)
    cond_S = None

    """
    Operators needed for case 1:
    a) M = Ψ* D Ψ -----> weights1
    b) Dual norm: Ψ* D^(-1) Ψ -----> dual_norm1
    """
    
    print("  Computing D^(-1)...")    
    D_inv_1_2D = dual_norm1(D_2D, Psi_2D)
    D_inv_1 = torch.kron(D_inv_1_2D, D_inv_1_2D)  # Full 1D diagonal operator

    print("  Computing M = Ψ* D Ψ ...")
    M_2D = weights1(D_2D, Psi_2D)
    
    Mherm_2D = 0.5 * (M_2D + M_2D.mH)

    jitter_scale = 1e-6
    jitter = float(jitter_scale) * Mherm_2D.abs().mean()
    Mherm_2D = Mherm_2D + jitter * torch.eye(H, device=Mherm_2D.device, dtype=Mherm_2D.dtype)
    torch.cuda.empty_cache()
    gc.collect()

    Mherm_2D = Mherm_2D.to(device)
    print('Calculating eigendecomposition...')  # Keep on CPU for stability in solves
    
    # --- Cache: eigendecomposition if possible (fast path) ---
    use_eig = False
    try:
        M64 = Mherm_2D.to(torch.complex64) if Mherm_2D.is_complex() else Mherm_2D.to(torch.float64)
        mu64, U64 = torch.linalg.eigh(M64.cpu())  # CPU is more robust
        mu_M = mu64.real.clamp_min(0.0).to(torch.float32).to(device)  # (n,) 
        U_M = U64.to(Mherm_2D.dtype).to(device)                         # (n,n) 
        use_eig = True
    except Exception as e:
        print("Warning: eigh failed; using solve-based projection fallback.", repr(e))
        use_eig = False

    # --- eps_scale via (clipped) geometric mean of eigenvalues ---
    # Requires self.mu_M if use_eig=True. If use_eig=False, we fall back to slogdet.
    if use_eig:
        mu = mu_M.to(device)  # (n,)
        mu_floor = 1e-8  # tune: 1e-10..1e-6 depending on conditioning
        mu_safe = mu.clamp_min(mu_floor)
        eps_scale = torch.exp((torch.log(mu_safe**2).mean())/ (2 * n)).item()
    else:
        # fallback: still can be unstable for near-singular M
        logdet = torch.slogdet(Mherm_2D).logabsdet
        eps_scale = torch.exp(logdet / 2 * ((n))).item()

    eps_scale = math.sqrt((2 * n)/(math.e * math.pi)) * math.sqrt(math.pi * n) ** (1/n) * eps_scale

    M = torch.kron(M_2D, M_2D)  # Full 1D operator
    M_herm = torch.kron(Mherm_2D, Mherm_2D)
    mu_M = torch.kron(mu_M, mu_M) if use_eig else print("  Warning: mu_M not available without eigendecomposition.")
    U_M = torch.kron(U_M, U_M) if use_eig else print("  Warning: U_M not available without eigendecomposition.")

    print("  Operators ready!")
    
    return Psi_2D, Psi_plus_2D, D_inv_1, M, eps_scale, mu_M, U_M, M_herm, cond_S, use_eig


# ============================================================================
# Transferability Evaluation
# ============================================================================

def generate_adversarial_examples(
    source_model: nn.Module,
    source_name: str,
    attacker: DGFPGDAttack,
    testloader: DataLoader,
    device: str,
    case,
    verbose: bool = False,
    args=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples using source model
    
    Returns:
        x_clean: Clean images
        x_adv: Adversarial images
        y_true: True labels
    """
    print(f"\nGenerating adversarial examples using {source_name}...")
    
    source_model.eval()

    # ------------------------------------------------------------------ #
    # Case 4: AutoAttack                                                   #
    # ------------------------------------------------------------------ #
    if case == "case4":
        try:
            from autoattack import AutoAttack
        except ImportError:
            raise ImportError(
                "autoattack is not installed. Install with: pip install autoattack"
            )

        # Collect the full dataset first (AutoAttack works on tensors, not loaders)
        all_clean, all_labels = [], []
        for x, y in tqdm(testloader, desc=f"Collecting data for {source_name}"):
            all_clean.append(x)
            all_labels.append(y)
        x_clean = torch.cat(all_clean, dim=0)
        y_true = torch.cat(all_labels, dim=0)

        aa_norm = getattr(args, "aa_norm", "Linf") if args is not None else "Linf"
        aa_version = getattr(args, "aa_version", "standard") if args is not None else "standard"
        epsilon = getattr(args, "epsilon", 8 / 255) if args is not None else 8 / 255
        batch_size = getattr(args, "batch_size", 64) if args is not None else 64

        print(
            f"  Running AutoAttack ({aa_version}, norm={aa_norm}, ε={epsilon:.4f}) "
            f"on {len(x_clean)} samples..."
        )
        adversary = AutoAttack(
            source_model,
            norm=aa_norm,
            eps=epsilon,
            version=aa_version,
            device=device,
            verbose=verbose,
        )
        x_adv = adversary.run_standard_evaluation(
            x_clean.to(device), y_true.to(device), bs=batch_size
        ).cpu()

        print(f"Generated {len(x_clean)} adversarial examples (AutoAttack)")
        return x_clean, x_adv, y_true

    # Update attacker's model to source model
    attacker.model = source_model
    
    all_clean = []
    all_adv = []
    all_labels = []
    
    source_model.eval()
    
    for batch_idx, (x, y) in enumerate(tqdm(testloader, desc=f"Source: {source_name}")):
        x, y = x.to(device), y.to(device)
        
        # Generate adversarial examples
        if case == 'case1':
            x_adv, _, _ = attacker(x, y, random_init=True)
        else:
            x_adv, _ = attacker(x, y, random_init=True)

        all_clean.append(x.cpu())
        all_adv.append(x_adv.cpu())
        all_labels.append(y.cpu())
    
    x_clean = torch.cat(all_clean, dim=0)
    x_adv = torch.cat(all_adv, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    
    print(f"Generated {len(x_clean)} adversarial examples")
    
    return x_clean, x_adv, y_true


def evaluate_transferability(
    target_model: nn.Module,
    target_name: str,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    y_true: torch.Tensor,
    metrics_evaluator: AdversarialMetrics,
    device: str,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluate adversarial examples on target model
    
    Returns:
        Dictionary with transferability metrics
    """
    target_model.eval()
    
    all_metrics = {
        'clean_accuracy': [],
        'adversarial_accuracy': [],
        'attack_success_rate': [],
        'mean_l2_norm': [],
        'mean_linf_norm': [],
    }
    
    # Process in batches
    num_samples = len(x_clean)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        x_clean_batch = x_clean[start_idx:end_idx].to(device)
        x_adv_batch = x_adv[start_idx:end_idx].to(device)
        y_batch = y_true[start_idx:end_idx].to(device)
        
        # Compute metrics
        batch_metrics = metrics_evaluator.compute_all_metrics(
            target_model, x_clean_batch, x_adv_batch, y_batch
        )
        
        # Accumulate
        for key in all_metrics.keys():
            if key in batch_metrics and batch_metrics[key] is not None:
                all_metrics[key].append(batch_metrics[key])
    
    # Average metrics across batches
    final_metrics = {}
    for key, values in all_metrics.items():
        if values:
            final_metrics[key] = np.mean(values)
        else:
            final_metrics[key] = None
    
    return final_metrics


def evaluate_all_transferability(
    models_dict: Dict[str, nn.Module],
    attacker: DGFPGDAttack, # DGFPGDAttack | None (None for case4/AutoAttack)
    metrics_evaluator: AdversarialMetrics,
    testloader: DataLoader,
    args
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate transferability for all source-target model pairs
    
    Returns:
        Nested dictionary: results[source_model][target_model] = metrics
    """
    results = {}
    model_names = list(models_dict.keys())
    
    print("\n" + "=" * 80)
    print("TRANSFERABILITY EVALUATION".center(80))
    print("=" * 80)
    
    # For each source model
    for source_name in model_names:
        source_model = models_dict[source_name]
        results[source_name] = {}
        
        print(f"\n{'='*80}")
        print(f"SOURCE MODEL: {source_name}".center(80))
        print(f"{'='*80}")
        
        # Generate adversarial examples using source model
        x_clean, x_adv, y_true = generate_adversarial_examples(
            source_model, source_name, attacker, testloader, 
            args.device, args.case, args.verbose, args=args,
        )
        
        # Evaluate on all target models (including source itself)
        print(f"\nEvaluating on target models...")
        for target_name in model_names:
            target_model = models_dict[target_name]
            
            print(f"\n  Target: {target_name}")
            
            # Compute transferability metrics
            transfer_metrics = evaluate_transferability(
                target_model, target_name, x_clean, x_adv, y_true,
                metrics_evaluator, args.device, args.batch_size
            )
            
            results[source_name][target_name] = transfer_metrics
            
            # Print key metrics
            if target_name == source_name:
                print(f"    [SELF] ASR: {transfer_metrics['attack_success_rate']*100:.2f}%, "
                      f"Adv Acc: {transfer_metrics['adversarial_accuracy']*100:.2f}%")
            else:
                print(f"    Transfer ASR: {transfer_metrics['attack_success_rate']*100:.2f}%, "
                      f"Adv Acc: {transfer_metrics['adversarial_accuracy']*100:.2f}%")
    
    return results


# ============================================================================
# Results Visualization and Saving
# ============================================================================

def create_transferability_matrices(results: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create transferability matrices for ASR and accuracy
    
    Returns:
        asr_matrix: Attack success rate matrix (source x target)
        acc_matrix: Adversarial accuracy matrix (source x target)
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    
    asr_matrix = np.zeros((n_models, n_models))
    acc_matrix = np.zeros((n_models, n_models))
    
    for i, source in enumerate(model_names):
        for j, target in enumerate(model_names):
            metrics = results[source][target]
            asr_matrix[i, j] = metrics['attack_success_rate'] * 100
            acc_matrix[i, j] = metrics['adversarial_accuracy'] * 100
    
    return asr_matrix, acc_matrix


def plot_transferability_heatmap(
    matrix: np.ndarray,
    model_names: List[str],
    title: str,
    output_path: str,
    cmap: str = 'RdYlGn_r',
    fmt: str = '.1f'
):
    """Plot and save transferability heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=model_names,
        yticklabels=model_names,
        cbar_kws={'label': title},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Target Model', fontsize=12, fontweight='bold')
    plt.ylabel('Source Model', fontsize=12, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap: {output_path}")


def print_transferability_table(results: Dict, args):
    """Print formatted transferability results table"""
    model_names = list(results.keys())
    
    print("\n" + "=" * 140)
    print("TRANSFERABILITY RESULTS".center(140))
    print("=" * 140)
    
    # Print ASR table
    print("\nAttack Success Rate (%) - Rows: Source, Columns: Target")
    print("-" * 140)
    
    # Header
    header = f"{'Source vs Target':<20}"
    for target in model_names:
        header += f"{target[:15]:>16}"
    print(header)
    print("-" * 140)
    
    # Rows
    for source in model_names:
        row = f"{source[:20]:<20}"
        for target in model_names:
            asr = results[source][target]['attack_success_rate'] * 100
            if source == target:
                row += f"{asr:>15.2f}*"  # Mark diagonal (self-attack)
            else:
                row += f"{asr:>16.2f}"
        print(row)
    
    print("\n* Diagonal values are self-attack success rates")
    
    # Print accuracy table
    print("\n" + "-" * 140)
    print("Adversarial Accuracy (%) - Rows: Source, Columns: Target")
    print("-" * 140)
    
    # Header
    header = f"{'Source vs Target':<20}"
    for target in model_names:
        header += f"{target[:15]:>16}"
    print(header)
    print("-" * 140)
    
    # Rows
    for source in model_names:
        row = f"{source[:20]:<20}"
        for target in model_names:
            acc = results[source][target]['adversarial_accuracy'] * 100
            if source == target:
                row += f"{acc:>15.2f}*"
            else:
                row += f"{acc:>16.2f}"
        print(row)
    
    print("\n* Diagonal values are self-attack robustness")
    print("=" * 140)


def compute_transferability_statistics(results: Dict) -> Dict:
    """Compute aggregate transferability statistics"""
    model_names = list(results.keys())
    stats = {}
    
    for source in model_names:
        # Get transfer ASR (excluding self-attack)
        transfer_asrs = []
        for target in model_names:
            if target != source:
                transfer_asrs.append(results[source][target]['attack_success_rate'] * 100)
        
        stats[source] = {
            'self_asr': results[source][source]['attack_success_rate'] * 100,
            'mean_transfer_asr': np.mean(transfer_asrs),
            'std_transfer_asr': np.std(transfer_asrs),
            'max_transfer_asr': np.max(transfer_asrs),
            'min_transfer_asr': np.min(transfer_asrs),
        }
    
    return stats


def save_results(results: Dict, args):
    """Save all results to files"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_names = list(results.keys())
    
    # 1. Save detailed JSON
    json_file = os.path.join(args.output_dir, f'{args.case}_transferability_detailed.json')
    serializable = {
        source: {
            target: {k: float(v) if v is not None else None for k, v in metrics.items()}
            for target, metrics in targets.items()
        }
        for source, targets in results.items()
    }
    with open(json_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✓ Detailed results saved to {json_file}")
    
    # 2. Save summary statistics
    stats = compute_transferability_statistics(results)
    stats_file = os.path.join(args.output_dir, f'{args.case}_transferability_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Transferability Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<25} {'Self ASR':>10} {'Mean Transfer':>15} {'Std':>10} "
                f"{'Max':>10} {'Min':>10}\n")
        f.write("-" * 80 + "\n")
        
        for model, stat in stats.items():
            f.write(f"{model:<25} {stat['self_asr']:>9.2f}% "
                   f"{stat['mean_transfer_asr']:>14.2f}% {stat['std_transfer_asr']:>9.2f}% "
                   f"{stat['max_transfer_asr']:>9.2f}% {stat['min_transfer_asr']:>9.2f}%\n")
    
    print(f"✓ Statistics saved to {stats_file}")
    
    # 3. Save CSV matrices
    asr_matrix, acc_matrix = create_transferability_matrices(results)
    
    # ASR matrix
    asr_df = pd.DataFrame(asr_matrix, index=model_names, columns=model_names)
    asr_csv = os.path.join(args.output_dir, f'{args.case}_transferability_asr.csv')
    asr_df.to_csv(asr_csv)
    print(f"✓ ASR matrix saved to {asr_csv}")
    
    # Accuracy matrix
    acc_df = pd.DataFrame(acc_matrix, index=model_names, columns=model_names)
    acc_csv = os.path.join(args.output_dir, f'{args.case}_transferability_accuracy.csv')
    acc_df.to_csv(acc_csv)
    print(f"✓ Accuracy matrix saved to {acc_csv}")
    
    # 4. Save heatmaps if requested
    if args.save_heatmaps:
        # ASR heatmap
        asr_heatmap = os.path.join(args.output_dir, 
                                   f'{args.case}_transferability_asr_heatmap.png')
        plot_transferability_heatmap(
            asr_matrix, model_names,
            f'Attack Success Rate (%) - {args.case.upper()}',
            asr_heatmap,
            cmap='RdYlGn_r'
        )
        
        # Accuracy heatmap
        acc_heatmap = os.path.join(args.output_dir,
                                   f'{args.case}_transferability_accuracy_heatmap.png')
        plot_transferability_heatmap(
            acc_matrix, model_names,
            f'Adversarial Accuracy (%) - {args.case.upper()}',
            acc_heatmap,
            cmap='RdYlGn'
        )


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution"""
    args = parse_args()
     
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 80)
    print("CIFAR10 TRANSFERABILITY EVALUATION".center(80))
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Model Source: {args.model_source}")
    print(f"  Case: {args.case}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Attack: ε={args.epsilon:.4f}, γ={args.gamma:.4f}, K={args.num_steps}, τ={args.tau}")
    
    # Load data
    testloader = load_cifar10(args)
    
    # Load models
    models_dict = load_cifar10_models(args)
    
    if len(models_dict) < 2:
        print("\nWARNING: Transferability evaluation requires at least 2 models!")
        print("Please load more models or adjust --models argument")
        return
    
    case = args.case

    if case == "case4":
        print("\nCase 4 selected: Using AutoAttack for adversarial example generation.")
        print("No Gabor operators needed for Case 4.")
        attacker = None  # Not used for AutoAttack
        metrics_evaluator = AdversarialMetrics(
                device=args.device,
                lpips_net=args.lpips_net,
                verbose=args.verbose,
                M=None,
                tau=args.tau,
                case_type=4,
            )
    else:   
    # Generate Gabor operators (needed for Case 1)
        print("\nGenerating Gabor operators...")
        epsilon = args.epsilon
        tau = args.tau
        a = args.a
        b = args.b
        rho = args.rho
        window = args.window_type
        
        # Generate Gabor operators
        Psi_2D, Psi_plus_2D, D_inv_1, M, eps_scale, mu_M, U_M, M_herm, cond_S, use_eig \
            = generate_gabor_operators_cifar10(args.device, a, b, rho, window)
        
        # Initialize attacker
        print("\nInitializing DGF-PGD attacker...")
        attacker = DGFPGDAttack(
            model=list(models_dict.values())[0],
            loss_fn=nn.CrossEntropyLoss(),
            Psi_2D=Psi_2D, Psi_plus_2D=Psi_plus_2D,
            D_inv_1=D_inv_1,
            M=M, eps_scale=eps_scale, mu_M=mu_M, U_M=U_M, M_herm=M_herm, use_eig=use_eig,
            image_shape=(3, 32, 32),
            tau=tau, rho=rho, epsilon=epsilon, gamma=args.gamma,
            num_steps=args.num_steps, case=args.case,
            device=args.device, verbose=args.verbose
        )
        
        # Initialize metrics evaluator
        print("Initializing metrics evaluator...")
        metrics_evaluator = AdversarialMetrics(
            device=args.device,
            lpips_net=args.lpips_net,
            verbose=args.verbose,
            M=M,
            tau=args.tau,
            case_type=args.case
        )
        
    # Evaluate transferability
    results = evaluate_all_transferability(
        models_dict, attacker, metrics_evaluator, testloader, args
    )
    
    # Print and save results
    print_transferability_table(results, args)
    save_results(results, args)
    
    # Print summary statistics
    stats = compute_transferability_statistics(results)
    print("\n" + "=" * 80)
    print("TRANSFERABILITY SUMMARY".center(80))
    print("=" * 80)
    print(f"\n{'Model':<25} {'Self ASR':>10} {'Mean Transfer ASR':>20} {'Std':>10}")
    print("-" * 80)
    for model, stat in stats.items():
        print(f"{model:<25} {stat['self_asr']:>9.2f}% "
              f"{stat['mean_transfer_asr']:>19.2f}% {stat['std_transfer_asr']:>9.2f}%")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()