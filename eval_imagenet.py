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
from typing import Dict, List

from transforms import *
from dgf_pgd import DGFPGDAttack
from evaluation_metrics import AdversarialMetrics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pytorch_msssim import ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="ImageNet DGF-PGD Attack Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--case",
        type=str,
        default="case1",
        choices=["case1", "case2", "case3", "case4"],
        help="Attack case to evaluate",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/imagenet",
        help="ImageNet data directory (parent of val/)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of test samples"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Data loading workers"
    )

    parser.add_argument("--epsilon", type=float, default=4 / 255, help="Attack epsilon")
    parser.add_argument("--gamma", type=float, default=1.0, help="Step size")
    parser.add_argument("--num-steps", type=int, default=20, help="PGD iterations")
    parser.add_argument(
        "--parameter", type=float, default=0.1, help="Parameter"
    )
    parser.add_argument("--a", type=int, default=1, help="Time lattice parameter")
    parser.add_argument(
        "--b", type=int, default=112, help="Frequency lattice parameter"
    )
    parser.add_argument("--rho", type=float, default=1.0, help="Frame potential factor")
    parser.add_argument(
        "--window-type",
        type=str,
        default="Hann",
        choices=["Hann", "Blackman", "Gaussian"],
        help="Type of Gabor window function",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (default: the ImageNet transferability set)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory where RobustBench model weights are stored",
    )

    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"]
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save clean and adversarial images",
    )
    parser.add_argument(
        "--num-images", type=int, default=10, help="Number of image pairs to save"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./results/images",
        help="Directory to save images",
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

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
    priority = {"Linf": 0, "L2": 1, "corruptions": 2, "common": 3}
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
        assert mu_M_2D is not None
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


def evaluate_model(
    Psi_2D,
    Psi_plus_2D,
    model,
    model_name,
    attacker,
    metrics_evaluator,
    dataloader,
    run_case,
    args,
):
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    results = {}
    attacker.model = model
    images_saved = False

    if run_case == "case1":
        print(f"\nCase 1: Soft-thresholded Frame Attack")
        print("-" * 80)
        attacker.case = "case1"

        case1_metrics = []
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 1", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, eps_dgf, last_delta = attacker(images, labels, random_init=True)
            metrics = metrics_evaluator.compute_all_metrics_gabor(
                model, images, x_adv, labels, eps_dgf, last_delta
            )
            case1_metrics.append(metrics)

            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    pred_clean = model(images).argmax(dim=1)
                    pred_adv = model(x_adv).argmax(dim=1)

                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case1",
                    model_name,
                    args.num_images,
                )
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case1",
                    model_name,
                    args.num_images,
                )

                B, C, n, _ = images.shape
                N = Psi_2D.shape[0]
                Psi_2D_c = Psi_2D.to(torch.complex128).to(args.device)
                Psi_bc = Psi_2D_c.view(1, 1, N, n)
                PsiT_bc = Psi_2D_c.t().view(1, 1, n, N)
                z = torch.matmul(Psi_bc, images.to(dtype=Psi_2D_c.dtype))
                z = torch.matmul(z, PsiT_bc)
                z = distord(z, args.parameter)
                Psi_plus_2D_c = Psi_plus_2D.to(torch.complex128).to(args.device)
                Psi_plus_bc = Psi_plus_2D_c.view(1, 1, n, N)
                Psi_plusT_bc = Psi_plus_2D_c.t().view(1, 1, N, n)
                x_tilde = torch.matmul(Psi_plus_bc, z)
                x_tilde = torch.matmul(x_tilde, Psi_plusT_bc).real
                x_tilde = torch.clamp(x_tilde, 0.0, 1.0)

                try:
                    save_gabor_spectrograms(
                        images,
                        x_adv,
                        last_delta,
                        labels,
                        Psi_2D,
                        args.image_dir,
                        "case1",
                        model_name,
                        args.num_images,
                        x_tilde=x_tilde,
                    )
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")

                images_saved = True

        results["case1"] = aggregate_metrics(case1_metrics)
        print_summary(results["case1"], "Case 1")

    elif run_case == "case2":
        print(f"\nCase 2: L2 PGD Attack")
        print("-" * 80)
        attacker.case = "case2"

        case2_metrics = []
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 2", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, last_delta = attacker(images, labels, random_init=True)
            metrics = metrics_evaluator.compute_all_metrics(
                model, images, x_adv, labels
            )
            case2_metrics.append(metrics)

            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    pred_clean = model(images).argmax(dim=1)
                    pred_adv = model(x_adv).argmax(dim=1)

                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case2",
                    model_name,
                    args.num_images,
                )
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case2",
                    model_name,
                    args.num_images,
                )
                save_gabor_spectrograms(
                    images,
                    x_adv,
                    last_delta,
                    labels,
                    Psi_2D,
                    args.image_dir,
                    "case2",
                    model_name,
                    args.num_images,
                    x_tilde=None,
                )
                images_saved = True

        results["case2"] = aggregate_metrics(case2_metrics)
        print_summary(results["case2"], "Case 2")

    elif run_case == "case3":
        print(f"\nCase 3: Fourier-based PGD Attack")
        print("-" * 80)
        attacker.case = "case3"

        case3_metrics = []
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 3", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, last_delta = attacker(images, labels, random_init=False)
            metrics = metrics_evaluator.compute_all_metrics(
                model, images, x_adv, labels
            )
            case3_metrics.append(metrics)

            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    pred_clean = model(images).argmax(dim=1)
                    pred_adv = model(x_adv).argmax(dim=1)

                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case3",
                    model_name,
                    args.num_images,
                )
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case3",
                    model_name,
                    args.num_images,
                )
                save_gabor_spectrograms(
                    images,
                    x_adv,
                    last_delta,
                    labels,
                    Psi_2D,
                    args.image_dir,
                    "case3",
                    model_name,
                    args.num_images,
                    x_tilde=None,
                )
                images_saved = True

        results["case3"] = aggregate_metrics(case3_metrics)
        print_summary(results["case3"], "Case 3")

    elif run_case == "case4":
        print(f"\nCase 4: High-frequency PGD Attack")
        print("-" * 80)
        attacker.case = "case4"

        case4_metrics = []
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 4", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, last_delta = attacker(images, labels, random_init=False)
            metrics = metrics_evaluator.compute_all_metrics(
                model, images, x_adv, labels
            )
            case4_metrics.append(metrics)

            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    pred_clean = model(images).argmax(dim=1)
                    pred_adv = model(x_adv).argmax(dim=1)

                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case4",
                    model_name,
                    args.num_images,
                )
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case4",
                    model_name,
                    args.num_images,
                )
                save_gabor_spectrograms(
                    images,
                    x_adv,
                    last_delta,
                    labels,
                    Psi_2D,
                    args.image_dir,
                    "case4",
                    model_name,
                    args.num_images,
                    x_tilde=None,
                )
                images_saved = True

        results["case4"] = aggregate_metrics(case4_metrics)
        print_summary(results["case4"], "Case 4")

    return results


def save_image_comparison(
    clean_images,
    adv_images,
    labels,
    predictions_clean,
    predictions_adv,
    save_dir,
    case_name,
    model_name,
    num_images=10,
):
    os.makedirs(save_dir, exist_ok=True)

    num_images = min(num_images, clean_images.shape[0])

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        clean_img = np.clip(clean_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        adv_img = np.clip(adv_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)

        true_label = labels[i].item()
        pred_clean = predictions_clean[i].item()
        pred_adv = predictions_adv[i].item()

        axes[i, 0].imshow(clean_img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"Clean\nTrue: {true_label}\nPred: {pred_clean}", fontsize=10
        )

        axes[i, 1].imshow(adv_img)
        axes[i, 1].axis("off")
        color = "red" if pred_adv != true_label else "green"
        axes[i, 1].set_title(
            f"Adversarial\nTrue: {true_label}\nPred: {pred_adv}",
            fontsize=10,
            color=color,
        )

    plt.tight_layout()
    filepath = os.path.join(save_dir, f"{model_name}_{case_name}_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison images to: {filepath}")


def save_individual_images(
    clean_images, adv_images, labels, save_dir, case_name, model_name, num_images=10
):
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(num_images, clean_images.shape[0])

    for i in range(num_images):
        true_label = labels[i].item()

        clean_img = np.clip(clean_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        plt.figure(figsize=(3, 3))
        plt.imshow(clean_img)
        plt.axis("off")
        plt.title(f"{true_label}", fontsize=12)
        plt.savefig(
            os.path.join(
                save_dir, f"{model_name}_{case_name}_clean_{i:03d}_{true_label}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        adv_img = np.clip(adv_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        plt.figure(figsize=(3, 3))
        plt.imshow(adv_img)
        plt.axis("off")
        plt.title(f"{true_label} (adv)", fontsize=12)
        plt.savefig(
            os.path.join(
                save_dir, f"{model_name}_{case_name}_adv_{i:03d}_{true_label}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print(
        f"  Saved {num_images} clean and {num_images} adversarial images to: {save_dir}"
    )


def save_gabor_spectrograms(
    clean_images,
    adv_images,
    delta,
    labels,
    Psi_2D,
    save_dir,
    case_name,
    model_name,
    num_images=10,
    x_tilde=None,
):
    if delta.dim() == 1:
        B, C, H, W = clean_images.shape
        delta = delta.reshape(B, C, H, W)
    elif delta.dim() == 2:
        B, C, H, W = clean_images.shape
        delta = delta.reshape(B, C, H, W)

    os.makedirs(save_dir, exist_ok=True)

    num_images = min(num_images, clean_images.shape[0])
    B, C, H, W = clean_images.shape

    Psi_2D = Psi_2D.to(clean_images.device)

    for idx in range(num_images):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        true_label = labels[idx].item()

        clean_img = np.clip(clean_images[idx].cpu().permute(1, 2, 0).numpy(), 0, 1)
        adv_img = np.clip(adv_images[idx].cpu().permute(1, 2, 0).numpy(), 0, 1)
        perturbation = delta[idx].cpu().permute(1, 2, 0).numpy()

        ssim_value = None
        if SSIM_AVAILABLE:
            try:
                with torch.no_grad():
                    ssim_value = ssim(
                        clean_images[idx : idx + 1],
                        adv_images[idx : idx + 1],
                        data_range=1.0,
                        size_average=True,
                    ).item()
            except Exception:
                ssim_value = None

        axes[0, 0].imshow(clean_img)
        axes[0, 0].set_title(f"Clean image\n{true_label}", fontsize=18)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(adv_img)
        if ssim_value is not None:
            axes[0, 1].set_title(
                f"x_adv\n{true_label}\nSSIM: {ssim_value:.4f}", fontsize=18
            )
        else:
            axes[0, 1].set_title(f"x_adv\n{true_label}", fontsize=18)
        axes[0, 1].axis("off")

        pert_display = perturbation - perturbation.min()
        pert_display = pert_display / (pert_display.max() + 1e-10)
        axes[0, 2].imshow(pert_display)

        case_titles = {
            "case1": "Proposed attack δ\n",
            "case2": "Standard PGD attack δ\n",
            "case3": "Fourier-based PGD attack δ\n",
            "case4": "High-frequency PGD attack δ\n",
        }
        axes[0, 2].set_title(
            case_titles.get(case_name, f"{case_name} δ\n"), fontsize=18
        )
        axes[0, 2].axis("off")

        N = Psi_2D.shape[0]
        Psi_bc = Psi_2D.view(1, 1, N, H)
        PsiT_bc = Psi_2D.t().view(1, 1, H, N)

        def gabor2d_avg_magnitude(x_chw):
            x_chw = x_chw.unsqueeze(0).to(dtype=Psi_2D.dtype)
            tmp = torch.matmul(Psi_bc, x_chw)
            w = torch.matmul(tmp, PsiT_bc)
            return torch.abs(w).mean(dim=(0, 1)).detach().cpu().numpy()

        clean_gabor_2d = gabor2d_avg_magnitude(clean_images[idx])
        adv_gabor_2d = gabor2d_avg_magnitude(adv_images[idx])
        delta_gabor_2d = gabor2d_avg_magnitude(delta[idx])

        def compute_psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float("inf")
            max_val = max(img1.max(), img2.max())
            if max_val == 0:
                return float("inf")
            return 20 * np.log10(max_val / np.sqrt(mse))

        psnr_adv = compute_psnr(clean_gabor_2d, adv_gabor_2d)
        psnr_delta = compute_psnr(clean_gabor_2d, delta_gabor_2d)

        im1 = axes[1, 0].imshow(clean_gabor_2d, cmap="plasma", aspect="auto")
        axes[1, 0].set_title(f"PSNR: ∞ dB", fontsize=18)
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        im3 = axes[1, 1].imshow(adv_gabor_2d, cmap="plasma", aspect="auto")
        axes[1, 1].set_title(f"PSNR: {psnr_adv:.4f} dB", fontsize=18)
        axes[1, 1].axis("off")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        im4 = axes[1, 2].imshow(delta_gabor_2d, cmap="plasma", aspect="auto")
        axes[1, 2].set_title(f"PSNR: {psnr_delta:.4f} dB", fontsize=18)
        axes[1, 2].axis("off")
        plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

        plt.tight_layout()
        filename = f"{model_name}_{case_name}_spectrogram_{idx:03d}_{true_label}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"✓ Saved {num_images} Gabor spectrograms to: {save_dir}")


def aggregate_metrics(metrics_list):
    aggregated = {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    for key in all_keys:
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        aggregated[key] = np.mean(values) if values else None

    return aggregated


def print_summary(metrics, case_name):
    print(f"\n{case_name} Summary:")
    print(f"  ASR:           {metrics['attack_success_rate']*100:>6.2f}%")
    print(f"  Clean Acc:     {metrics['clean_accuracy']*100:>6.2f}%")
    print(f"  Adv Acc:       {metrics['adversarial_accuracy']*100:>6.2f}%")
    print(
        f"  L2 Norm:       {metrics['mean_l2_norm']:>6.4f} ± {metrics.get('std_l2_norm', 0):>6.4f}"
    )
    print(
        f"  Linf Norm:     {metrics['mean_linf_norm']:>6.4f} ± {metrics.get('std_linf_norm', 0):>6.4f}"
    )

    if case_name == "Case 1":
        if metrics.get("mean_gabor_frame_norm") is not None:
            print(f"  Gabor ||w||_D:    {metrics['mean_gabor_frame_norm']:>6.8f}")
        if metrics.get("feasible_frac") is not None:
            print(f"  Feasibility region:   {metrics['feasible_frac']:>6.4f}")
    if metrics.get("lpips_mean") is not None:
        print(
            f"  LPIPS:         {metrics['lpips_mean']:>6.4f} ± {metrics.get('lpips_std', 0):>6.4f}"
        )
    if metrics.get("ssim_mean") is not None:
        print(
            f"  SSIM:          {metrics['ssim_mean']:>6.4f} ± {metrics.get('ssim_std', 0):>6.4f}"
        )


def print_results_table(all_results, args):
    print("\n" + "=" * 140)
    print("IMAGENET DGF-PGD ATTACK RESULTS".center(140))
    print("=" * 140)

    case_labels = {
        "case1": "Case 1: Soft-thresholded Frame Attack",
        "case2": "Case 2: L2 PGD Attack",
        "case3": "Case 3: Fourier-based PGD Attack",
        "case4": "Case 4: High-frequency PGD Attack",
    }

    case_name = args.case
    case_label = case_labels.get(case_name, case_name)

    print(f"\n{case_label}")
    print("-" * 140)
    print(
        f"{'Model':<30} {'ASR':>8} {'Clean':>8} {'Adv':>8} {'L2':>10} {'Linf':>10} "
        f"{'||w||_D':>10} {'Feasible region':>10} {'LPIPS':>10} {'SSIM':>10}"
    )
    print("-" * 140)

    for model_name, results in all_results.items():
        if case_name in results:
            r = results[case_name]
            gabor_norm_str = (
                f"{r['mean_gabor_frame_norm']:.8f}"
                if r.get("mean_gabor_frame_norm") is not None
                else "N/A"
            )
            gabor_feas_str = (
                f"{r['feasible_frac']:.4f}"
                if r.get("feasible_frac") is not None
                else "N/A"
            )
            lpips_str = f"{r['lpips_mean']:.4f}" if r.get("lpips_mean") else "N/A"
            ssim_str = f"{r['ssim_mean']:.4f}" if r.get("ssim_mean") else "N/A"

            print(
                f"{model_name:<30} "
                f"{r['attack_success_rate']*100:>7.2f}% "
                f"{r['clean_accuracy']*100:>7.2f}% "
                f"{r['adversarial_accuracy']*100:>7.2f}% "
                f"{r['mean_l2_norm']:>9.4f} "
                f"{r['mean_linf_norm']:>9.4f} "
                f"{gabor_norm_str:>9s} "
                f"{gabor_feas_str:>9s}"
                f"{lpips_str:>9s} "
                f"{ssim_str:>9s}"
            )

    print("\n" + "=" * 140)


def save_results(all_results, args):
    os.makedirs(args.output_dir, exist_ok=True)

    summary_file = os.path.join(args.output_dir, f"{args.case}_imagenet_results.txt")
    with open(summary_file, "w") as f:
        f.write("ImageNet DGF-PGD Attack Results\n")
        f.write("=" * 160 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epsilon: {args.epsilon:.4f}\n")
        f.write(f"  Gamma: {args.gamma:.4f}\n")
        f.write(f"  Steps: {args.num_steps}\n")
        f.write(f"  parameter: {args.parameter}\n")
        f.write(f"  Samples: {args.num_samples}\n\n")

        case_name = args.case
        f.write(f"\n{case_name}\n")
        f.write("-" * 160 + "\n")
        f.write(
            f"{'Model':<30} {'ASR':>8} {'Clean':>8} {'Adv':>8} {'L2':>10} {'Linf':>10} "
            f"{'||w||_D':>10} {'Feasible region':>10} {'LPIPS':>10} {'SSIM':>10} {'PSNR':>10}\n"
        )
        f.write("-" * 160 + "\n")

        for model_name, results in all_results.items():
            if case_name in results:
                r = results[case_name]
                gabor_norm_str = (
                    f"{r['mean_gabor_frame_norm']:.8f}"
                    if r.get("mean_gabor_frame_norm") is not None
                    else "N/A"
                )
                gabor_feas_str = (
                    f"{r['feasible_frac']:.4f}"
                    if r.get("feasible_frac") is not None
                    else "N/A"
                )
                lpips_str = f"{r['lpips_mean']:.4f}" if r.get("lpips_mean") else "N/A"
                ssim_str = f"{r['ssim_mean']:.4f}" if r.get("ssim_mean") else "N/A"
                psnr_str = f"{r['psnr_mean']:.2f}" if r.get("psnr_mean") else "N/A"

                f.write(
                    f"{model_name:<30} "
                    f"{r['attack_success_rate']*100:>7.2f}% "
                    f"{r['clean_accuracy']*100:>7.2f}% "
                    f"{r['adversarial_accuracy']*100:>7.2f}% "
                    f"{r['mean_l2_norm']:>9.4f} "
                    f"{r['mean_linf_norm']:>9.4f} "
                    f"{gabor_norm_str:>9s} "
                    f"{gabor_feas_str:>9s}"
                    f"{lpips_str:>9s} "
                    f"{ssim_str:>9s} "
                    f"{psnr_str:>9s}\n"
                )

    print(f"\n✓ Summary saved to {summary_file}")

    json_file = os.path.join(
        args.output_dir, f"{args.case}_imagenet_results_detailed.json"
    )
    serializable = {
        model: {
            case: {k: float(v) if v is not None else None for k, v in metrics.items()}
            for case, metrics in cases.items()
        }
        for model, cases in all_results.items()
    }
    with open(json_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"✓ Detailed results saved to {json_file}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 80)
    print("IMAGENET DGF-PGD ATTACK EVALUATION".center(80))
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Case: {args.case}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples: {args.num_samples}")
    print(
        f"  Attack: ε={args.epsilon:.4f}, γ={args.gamma:.4f}, K={args.num_steps}, τ={args.parameter}"
    )

    testloader = load_imagenet(args)
    models_dict = load_imagenet_models(args)

    a = args.a
    b = args.b
    window = args.window_type
    epsilon = args.epsilon
    parameter = args.parameter
    rho = args.rho

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
        gamma=args.gamma,
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

    all_results = {}
    for model_name, model in models_dict.items():
        results = evaluate_model(
            Psi_2D,
            Psi_plus_2D,
            model,
            model_name,
            attacker,
            metrics_evaluator,
            testloader,
            args.case,
            args,
        )
        all_results[model_name] = results
        print_results_table(all_results, args)
        save_results(all_results, args)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
