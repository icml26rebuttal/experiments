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
import re
import argparse
import json
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transforms import *
from dgf_pgd import DGFPGDAttack
from evaluation_metrics import AdversarialMetrics


DEFAULT_IMAGENET_MODELS = [
    "Amini2024MeanSparse_ConvNeXt-L",
    "Bai2024MixedNUTS",
    "Debenedetti2022Light_XCiT-M12",
    "Engstrom2019Robustness",
    "Liu2023Comprehensive_ConvNeXt-B",
    "RodriguezMunoz2024Characterizing_Swin-B",
    "Salman2020Do_R50",
    "Singh2023Revisiting_ViT-B-ConvStem",
    "Wong2020Fast",
]


def hparam_folder_name(a, b, window, epsilon, parameter, gamma):
    return f"a{a}_b{b}_{window}_eps{epsilon:.4f}_parameter{parameter}_gamma{gamma}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ImageNet Adversarial Attack Transferability Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--case",
        type=str,
        default="case1",
        choices=["case1", "case2", "case3", "case4"],
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/imagenet",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epsilon", type=float, default=4 / 255)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--parameter", type=float, default=0.1)
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--b", type=int, default=112)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument(
        "--window-type",
        type=str,
        default="Hann",
        choices=["Hann", "Blackman", "Gaussian"],
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/transferability",
    )
    parser.add_argument(
        "--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"]
    )
    parser.add_argument("--save-heatmaps", action="store_true")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def load_imagenet(args):
    print("\nLoading ImageNet dataset...")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    val_dir = os.path.join(args.data_root, "val")

    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet validation directory not found at {val_dir}. "
            "Set --data-root to the parent of the 'val/' folder."
        )

    testset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    print(f"  Found {len(testset)} images in {len(testset.classes)} classes")

    if args.num_samples is not None and args.num_samples < len(testset):
        indices = np.random.choice(len(testset), args.num_samples, replace=False)
        testset = Subset(testset, indices)

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Loaded {len(testset)} test samples")
    return testloader


def _canonical_model_name(weight_filename: str) -> str | None:
    if not weight_filename.endswith(".pt"):
        return None

    shard_match = re.match(r"^(?P<name>.+)\.pt_m\d+\.pt$", weight_filename)
    if shard_match:
        return shard_match.group("name")

    shard_match = re.match(r"^(?P<name>.+)_m\d+\.pt$", weight_filename)
    if shard_match:
        return shard_match.group("name")

    return weight_filename[:-3]


def _threat_model_priority(threat_model: str) -> tuple[int, str]:
    priority = {
        "Linf": 0,
        "L2": 1,
        "corruptions": 2,
        "common": 3,
    }
    return (priority.get(threat_model, 99), threat_model)


def load_imagenet_models(args):
    print("\nLoading ImageNet models from RobustBench...")

    try:
        from robustbench.utils import load_model as rb_load_model
        import robustbench.utils as _rb_utils
    except ImportError:
        print("\nERROR: robustbench not installed!")
        print("Install with: pip install robustbench")
        raise ImportError("robustbench is required for ImageNet models")

    try:
        import gdown

        def _download_gdrive_fixed(gdrive_id, fname_save):
            fname_save = str(fname_save)
            print(f"  Downloading via gdown: {fname_save} (gdrive_id={gdrive_id})")
            gdown.download(id=gdrive_id, output=fname_save, quiet=False)

        _rb_utils.download_gdrive = _download_gdrive_fixed
    except ImportError:
        print("  Warning: gdown not installed, using default downloader")

    desired_models = args.models or DEFAULT_IMAGENET_MODELS
    base_dir = os.path.join(args.models_dir, "imagenet")

    discovered: Dict[str, List[str]] = {}
    if os.path.isdir(base_dir):
        for threat_model in sorted(os.listdir(base_dir)):
            threat_dir = os.path.join(base_dir, threat_model)
            if not os.path.isdir(threat_dir):
                continue
            for filename in sorted(os.listdir(threat_dir)):
                model_name = _canonical_model_name(filename)
                if model_name is None:
                    continue

                file_path = os.path.join(threat_dir, filename)
                if os.path.getsize(file_path) < 100_000:
                    continue

                discovered.setdefault(model_name, []).append(threat_model)

    ordered_candidates = []
    missing_models = []
    for model_name in desired_models:
        threat_models = discovered.get(model_name, [])
        if not threat_models:
            missing_models.append(model_name)
            continue
        threat_model = sorted(set(threat_models), key=_threat_model_priority)[0]
        ordered_candidates.append((model_name, threat_model))

    if missing_models:
        print("\nWarning: these requested models were not found on disk:")
        for model_name in missing_models:
            print(f"  - {model_name}")
        print(f"  Looked under: {base_dir}")

    if not ordered_candidates:
        raise ValueError(
            f"No matching models found under {base_dir}. "
            "Download the requested RobustBench ImageNet weights first."
        )

    print(f"Loading {len(ordered_candidates)} models:")
    for model_name, threat_model in ordered_candidates:
        print(f"  - {model_name} (threat_model={threat_model})")

    models_dict = {}
    for model_name, threat_model in ordered_candidates:
        try:
            print(f"\n  Loading {model_name} [{threat_model}]...")
            model = rb_load_model(
                model_name=model_name,
                dataset="imagenet",
                threat_model=threat_model,
                model_dir=args.models_dir,
            ).to(args.device)
            models_dict[model_name] = model.eval()
            print("    ✓ Success")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    if not models_dict:
        raise RuntimeError("No models loaded successfully!")

    print(f"\n✓ Successfully loaded {len(models_dict)} models")
    return models_dict


def generate_gabor_operators(device, a, b, rho, image_size=224, window_type="Hann"):
    print(f"\nGenerating 2D Gabor operators for {image_size}x{image_size} images...")

    H, W = image_size, image_size
    n = H * W

    print(f"  Dimensions: H={H}, W={W}, N={(n) / (a * b)}")

    print("  Generating Ψ...")
    Psi_2D = DGT(H, a=a, b=b, window=window_type)
    Psi_2D = Psi_2D / torch.linalg.norm(Psi_2D, dim=1, keepdim=True)

    print("  Computing Ψ^+...")
    Psi_plus_2D = torch.linalg.pinv(Psi_2D)

    print("  Generating D...")
    D_2D = diag_weights_from_mc_row_sums(Psi_2D, mode="down")

    print("  Computing the condition number of the frame...")
    S_2D = frameop_DGT(Psi_2D)
    sv = torch.linalg.svdvals(S_2D)
    sv_pos = sv[sv > 1e-10]
    cond_S = (sv_pos[0] / sv_pos[-1]).item() if sv_pos.numel() > 0 else float("inf")
    if not torch.all(sv > 1e-10):
        print(
            f"  Warning: S is singular (rank {sv_pos.numel()}/{S_2D.shape[0]}), "
            f"cond={cond_S:.2f}"
        )

    print("  Computing D^(-1)...")
    D_inv_1_2D = dual_norm1(D_2D, Psi_2D)

    print("  Computing M = Ψ* D Ψ ...")
    M_2D = weights1(D_2D, Psi_2D)

    Mherm_2D = 0.5 * (M_2D + M_2D.mH)

    jitter_scale = 1e-6
    jitter = float(jitter_scale) * Mherm_2D.abs().mean()
    Mherm_2D = Mherm_2D + jitter * torch.eye(
        H, device=Mherm_2D.device, dtype=Mherm_2D.dtype
    )
    torch.cuda.empty_cache()
    gc.collect()

    Mherm_2D = Mherm_2D.to(device)
    print("Calculating eigendecomposition...")

    use_eig = False
    try:
        M64 = (
            Mherm_2D.to(torch.complex64)
            if Mherm_2D.is_complex()
            else Mherm_2D.to(torch.float64)
        )
        mu64, U64 = torch.linalg.eigh(M64.cpu())
        mu_M_2D = mu64.real.clamp_min(0.0).to(torch.float32).to(device)
        U_M_2D = U64.to(Mherm_2D.dtype).to(device)
        use_eig = True
    except Exception as e:
        print("Warning: eigh failed; using solve-based projection fallback.", repr(e))
        use_eig = False
        mu_M_2D = None
        U_M_2D = None

    if use_eig:
        print("  Computing eps_scale via eigendecomposition...")
        mu = mu_M_2D.to(device)
        mu_floor = 1e-8
        mu_safe = mu.clamp_min(mu_floor)
        eps_scale = torch.exp((torch.log(mu_safe**2).mean()) / (2 * n)).item()
    else:
        logdet = torch.slogdet(Mherm_2D).logabsdet
        eps_scale = torch.exp(logdet / 2 * n).item()

    eps_scale = (
        math.sqrt((2 * n) / (math.e * math.pi))
        * math.sqrt(math.pi * n) ** (1 / n)
        * eps_scale
    )

    print(f"  eps_scale = {eps_scale:.6g}")
    print("  Operators ready!")

    return (
        Psi_2D,
        Psi_plus_2D,
        D_inv_1_2D,
        M_2D,
        eps_scale,
        mu_M_2D,
        U_M_2D,
        Mherm_2D,
        cond_S,
        use_eig,
    )


def generate_adversarial_examples(
    source_model: nn.Module,
    source_name: str,
    attacker: DGFPGDAttack,
    testloader: DataLoader,
    device: str,
    case,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print(f"\nGenerating adversarial examples using {source_name}...")

    attacker.model = source_model
    all_clean = []
    all_adv = []
    all_labels = []

    source_model.eval()

    for batch_idx, (x, y) in enumerate(tqdm(testloader, desc=f"Source: {source_name}")):
        x, y = x.to(device), y.to(device)

        if case == "case1":
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
    batch_size: int = 64,
) -> Dict[str, float]:
    target_model.eval()

    all_metrics = {
        "clean_accuracy": [],
        "adversarial_accuracy": [],
        "attack_success_rate": [],
        "mean_l2_norm": [],
        "mean_linf_norm": [],
    }

    num_samples = len(x_clean)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        x_clean_batch = x_clean[start_idx:end_idx].to(device)
        x_adv_batch = x_adv[start_idx:end_idx].to(device)
        y_batch = y_true[start_idx:end_idx].to(device)

        batch_metrics = metrics_evaluator.compute_all_metrics(
            target_model, x_clean_batch, x_adv_batch, y_batch
        )

        for key in all_metrics.keys():
            if key in batch_metrics and batch_metrics[key] is not None:
                all_metrics[key].append(batch_metrics[key])

    final_metrics = {}
    for key, values in all_metrics.items():
        final_metrics[key] = np.mean(values) if values else None

    return final_metrics


def evaluate_all_transferability(
    models_dict: Dict[str, nn.Module],
    attacker: DGFPGDAttack,
    metrics_evaluator: AdversarialMetrics,
    testloader: DataLoader,
    args,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = {}
    model_names = list(models_dict.keys())

    print("\n" + "=" * 80)
    print("TRANSFERABILITY EVALUATION".center(80))
    print("=" * 80)

    for source_name in model_names:
        source_model = models_dict[source_name]
        results[source_name] = {}

        print(f"\n{'=' * 80}")
        print(f"SOURCE MODEL: {source_name}".center(80))
        print(f"{'=' * 80}")

        x_clean, x_adv, y_true = generate_adversarial_examples(
            source_model,
            source_name,
            attacker,
            testloader,
            args.device,
            args.case,
            args.verbose,
        )

        print(f"\nEvaluating on target models...")
        for target_name in model_names:
            target_model = models_dict[target_name]

            print(f"\n  Target: {target_name}")

            transfer_metrics = evaluate_transferability(
                target_model,
                target_name,
                x_clean,
                x_adv,
                y_true,
                metrics_evaluator,
                args.device,
                args.batch_size,
            )

            results[source_name][target_name] = transfer_metrics

            if target_name == source_name:
                print(
                    f"    [SELF] ASR: {transfer_metrics['attack_success_rate'] * 100:.2f}%, "
                    f"Adv Acc: {transfer_metrics['adversarial_accuracy'] * 100:.2f}%"
                )
            else:
                print(
                    f"    Transfer ASR: {transfer_metrics['attack_success_rate'] * 100:.2f}%, "
                    f"Adv Acc: {transfer_metrics['adversarial_accuracy'] * 100:.2f}%"
                )

    return results


def create_transferability_matrices(results: Dict) -> Tuple[np.ndarray, np.ndarray]:
    model_names = list(results.keys())
    n_models = len(model_names)

    asr_matrix = np.zeros((n_models, n_models))
    acc_matrix = np.zeros((n_models, n_models))

    for i, source in enumerate(model_names):
        for j, target in enumerate(model_names):
            metrics = results[source][target]
            asr_matrix[i, j] = metrics["attack_success_rate"] * 100
            acc_matrix[i, j] = metrics["adversarial_accuracy"] * 100

    return asr_matrix, acc_matrix


def plot_transferability_heatmap(
    matrix: np.ndarray,
    model_names: List[str],
    title: str,
    output_path: str,
    cmap: str = "RdYlGn_r",
    fmt: str = ".1f",
):
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=model_names,
        yticklabels=model_names,
        cbar_kws={"label": title},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Target Model", fontsize=12, fontweight="bold")
    plt.ylabel("Source Model", fontsize=12, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved heatmap: {output_path}")


def print_transferability_table(results: Dict, args):
    model_names = list(results.keys())

    print("\n" + "=" * 140)
    print("TRANSFERABILITY RESULTS".center(140))
    print("=" * 140)

    print("\nAttack Success Rate (%) - Rows: Source, Columns: Target")
    print("-" * 140)

    col_header = "Source \\ Target"
    header = f"{col_header:<20}"
    for target in model_names:
        header += f"{target[:15]:>16}"
    print(header)
    print("-" * 140)

    for source in model_names:
        row = f"{source[:20]:<20}"
        for target in model_names:
            asr = results[source][target]["attack_success_rate"] * 100
            if source == target:
                row += f"{asr:>15.2f}*"
            else:
                row += f"{asr:>16.2f}"
        print(row)

    print("\n* Diagonal values are self-attack success rates")

    print("\n" + "-" * 140)
    print("Adversarial Accuracy (%) - Rows: Source, Columns: Target")
    print("-" * 140)

    header = f"{col_header:<20}"
    for target in model_names:
        header += f"{target[:15]:>16}"
    print(header)
    print("-" * 140)

    for source in model_names:
        row = f"{source[:20]:<20}"
        for target in model_names:
            acc = results[source][target]["adversarial_accuracy"] * 100
            if source == target:
                row += f"{acc:>15.2f}*"
            else:
                row += f"{acc:>16.2f}"
        print(row)

    print("\n* Diagonal values are self-attack robustness")
    print("=" * 140)


def compute_transferability_statistics(results: Dict) -> Dict:
    model_names = list(results.keys())
    stats = {}

    for source in model_names:
        transfer_asrs = []
        for target in model_names:
            if target != source:
                transfer_asrs.append(
                    results[source][target]["attack_success_rate"] * 100
                )

        stats[source] = {
            "self_asr": results[source][source]["attack_success_rate"] * 100,
            "mean_transfer_asr": np.mean(transfer_asrs),
            "std_transfer_asr": np.std(transfer_asrs),
            "max_transfer_asr": np.max(transfer_asrs),
            "min_transfer_asr": np.min(transfer_asrs),
        }

    return stats


def save_results(results: Dict, args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_names = list(results.keys())

    json_file = os.path.join(
        args.output_dir, f"{args.case}_transferability_detailed.json"
    )
    serializable = {
        source: {
            target: {k: float(v) if v is not None else None for k, v in metrics.items()}
            for target, metrics in targets.items()
        }
        for source, targets in results.items()
    }
    with open(json_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✓ Detailed results saved to {json_file}")

    stats = compute_transferability_statistics(results)
    stats_file = os.path.join(args.output_dir, f"{args.case}_transferability_stats.txt")
    with open(stats_file, "w") as f:
        f.write("Transferability Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"{'Model':<25} {'Self ASR':>10} {'Mean Transfer':>15} {'Std':>10} "
            f"{'Max':>10} {'Min':>10}\n"
        )
        f.write("-" * 80 + "\n")

        for model, stat in stats.items():
            f.write(
                f"{model:<25} {stat['self_asr']:>9.2f}% "
                f"{stat['mean_transfer_asr']:>14.2f}% {stat['std_transfer_asr']:>9.2f}% "
                f"{stat['max_transfer_asr']:>9.2f}% {stat['min_transfer_asr']:>9.2f}%\n"
            )

    print(f"✓ Statistics saved to {stats_file}")

    asr_matrix, acc_matrix = create_transferability_matrices(results)

    asr_df = pd.DataFrame(asr_matrix, index=model_names, columns=model_names)
    asr_csv = os.path.join(args.output_dir, f"{args.case}_transferability_asr.csv")
    asr_df.to_csv(asr_csv)
    print(f"✓ ASR matrix saved to {asr_csv}")

    acc_df = pd.DataFrame(acc_matrix, index=model_names, columns=model_names)
    acc_csv = os.path.join(args.output_dir, f"{args.case}_transferability_accuracy.csv")
    acc_df.to_csv(acc_csv)
    print(f"✓ Accuracy matrix saved to {acc_csv}")

    if args.save_heatmaps:
        asr_heatmap = os.path.join(
            args.output_dir, f"{args.case}_transferability_asr_heatmap.png"
        )
        plot_transferability_heatmap(
            asr_matrix,
            model_names,
            f"Attack Success Rate (%) - {args.case.upper()}",
            asr_heatmap,
            cmap="RdYlGn_r",
        )

        acc_heatmap = os.path.join(
            args.output_dir, f"{args.case}_transferability_accuracy_heatmap.png"
        )
        plot_transferability_heatmap(
            acc_matrix,
            model_names,
            f"Adversarial Accuracy (%) - {args.case.upper()}",
            acc_heatmap,
            cmap="RdYlGn",
        )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 80)
    print("IMAGENET TRANSFERABILITY EVALUATION".center(80))
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Case: {args.case}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples: {args.num_samples}")
    print(
        f"  Attack: ε={args.epsilon:.4f}, γ={args.gamma:.4f}, K={args.num_steps}, τ={args.parameter}"
    )

    default_models_note = args.models or DEFAULT_IMAGENET_MODELS
    print(f"  Models: {', '.join(default_models_note)}")

    testloader = load_imagenet(args)
    models_dict = load_imagenet_models(args)

    if len(models_dict) < 2:
        print("\nWARNING: Transferability evaluation requires at least 2 models!")
        print("Please load more models or adjust --models argument")
        return

    a = args.a
    b = args.b
    window = args.window_type
    epsilon = args.epsilon
    parameter = args.parameter
    gamma = args.gamma

    if args.case == "case1":
        hp_name = hparam_folder_name(a, b, window, epsilon, parameter, gamma)
        print(f"\nCase 1 hparams: {hp_name}")
        print(
            f"  a={a}, b={b}, window={window}, eps={epsilon:.4f}, "
            f"parameter={parameter}, gamma={gamma}"
        )

    rho = args.rho

    print("\nGenerating Gabor operators...")
    (
        Psi_2D,
        Psi_plus_2D,
        D_inv_1_2D,
        M_2D,
        eps_scale,
        mu_M_2D,
        U_M_2D,
        Mherm_2D,
        cond_S,
        use_eig,
    ) = generate_gabor_operators(args.device, a, b, rho, 224, window)

    print("\nInitializing DGF-PGD attacker...")
    attacker = DGFPGDAttack(
        model=list(models_dict.values())[0],
        loss_fn=nn.CrossEntropyLoss(),
        Psi_2D=Psi_2D,
        Psi_plus_2D=Psi_plus_2D,
        D_inv_1=D_inv_1_2D,
        M=M_2D,
        eps_scale=eps_scale,
        mu_M=mu_M_2D,
        U_M=U_M_2D,
        M_herm=Mherm_2D,
        use_eig=use_eig,
        image_shape=(3, 224, 224),
        parameter=parameter,
        rho=rho,
        epsilon=epsilon,
        gamma=gamma,
        num_steps=args.num_steps,
        case=args.case,
        device=args.device,
        verbose=args.verbose,
    )

    case_num = int(args.case[-1])
    print("Initializing metrics evaluator...")
    metrics_evaluator = AdversarialMetrics(
        device=args.device,
        lpips_net=args.lpips_net,
        verbose=args.verbose,
        M=M_2D,
        parameter=parameter,
        case_type=case_num,
    )

    results = evaluate_all_transferability(
        models_dict, attacker, metrics_evaluator, testloader, args
    )

    print_transferability_table(results, args)
    save_results(results, args)

    stats = compute_transferability_statistics(results)
    print("\n" + "=" * 80)
    print("TRANSFERABILITY SUMMARY".center(80))
    print("=" * 80)
    print(f"\n{'Model':<25} {'Self ASR':>10} {'Mean Transfer ASR':>20} {'Std':>10}")
    print("-" * 80)
    for model, stat in stats.items():
        print(
            f"{model:<25} {stat['self_asr']:>9.2f}% "
            f"{stat['mean_transfer_asr']:>19.2f}% {stat['std_transfer_asr']:>9.2f}%"
        )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
