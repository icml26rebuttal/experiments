import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.fft
import torch.nn as nn
from typing import Callable, Optional, Tuple
from transforms import *


class DGFPGDAttack:

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        Psi_2D: torch.Tensor,
        Psi_plus_2D: torch.Tensor,
        D_inv_1: torch.Tensor,
        M: Optional[torch.Tensor],
        eps_scale: Optional[float],
        mu_M: Optional[torch.Tensor],
        U_M: Optional[torch.Tensor],
        M_herm: Optional[torch.Tensor],
        use_eig: Optional[bool],
        image_shape: Tuple[int, int, int] = (3, 64, 64),
        case1b: bool = True,
        parameter: float = 0.1,
        rho: float = 1.0,
        epsilon: float = 0.1,
        gamma: float = 0.01,
        num_steps: int = 10,
        case="case1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.parameter = parameter
        self.case1b = case1b
        self.rho = rho
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_steps = num_steps
        self.case = case
        self.device = device
        self.verbose = verbose

        self.Psi_2D = Psi_2D.to(device)
        self.Psi_plus_2D = Psi_plus_2D.to(device)

        self.M = M.to(device)
        self.D_inv_1 = D_inv_1.to(device)

        self.C, self.H, self.W = image_shape
        self.n = self.H * self.W
        self.N = self.Psi_2D.shape[0]

        self.proj_max_iter = 25
        self.proj_tol = 1e-5

        self.eps_scale = eps_scale
        self.mu_M = mu_M.to(device)
        self.U_M = U_M.to(device)
        self.Mherm = M_herm.to(device)
        self.use_eig = use_eig

        if self.verbose:
            print("DGF-PGD Attack (Case 1)")
            print(
                f"  Image shape: ({self.C},{self.H},{self.W}), n={self.n}, N={self.N}"
            )
            print(f"  eps={self.epsilon}, gamma={self.gamma}, steps={self.num_steps}")
            print(f"  projection: {'eig' if self.use_eig else 'solve-fallback'}")
            print(
                f"  eps_scale={self.eps_scale:.6g},\
                   eps_dgf={(self.eps_scale * self.epsilon):.6g}"
            )

    def attack_case1(
        self, x: torch.Tensor, y: torch.Tensor, random_init: bool = True
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        if (C, H, W) != (self.C, self.H, self.W):
            raise ValueError(
                f"Input shape mismatch: got ({C},{H},{W}) expected ({self.C},{self.H},{self.W})"
            )

        K = B * C
        n = self.n

        epsilon_dgf = self.eps_scale * self.epsilon

        is_2d_factorized = self.M.shape[0] == H

        n1 = self.Psi_2D.shape[1]
        Psi_bc = self.Psi_2D.view(1, 1, self.N, n1)
        PsiT_bc = self.Psi_2D.t().view(1, 1, n1, self.N)
        z = torch.matmul(Psi_bc, x.to(torch.complex64))
        z = torch.matmul(z, PsiT_bc)
        z = distord(z, self.parameter)

        Psi_plus_bc = self.Psi_plus_2D.view(1, 1, n1, self.N)
        Psi_plusT_bc = self.Psi_plus_2D.t().view(1, 1, self.N, n1)

        if self.case1b:
            x_tilde = x
        else:
            x_tilde = torch.matmul(Psi_plus_bc, z)
            x_tilde = torch.matmul(x_tilde, Psi_plusT_bc).real
            x_tilde = torch.clamp(x_tilde, 0.0, 1.0)

        if random_init:
            delta = torch.empty((B, C, H, W), device=self.device, dtype=x.dtype)
            delta.uniform_(-epsilon_dgf, epsilon_dgf)
            delta_flat = delta.reshape(K, n)
            if is_2d_factorized:
                d2 = delta_flat.reshape(K, H, W).to(self.M.dtype)
                MdMT = torch.einsum("ij,kjp,lp->kil", self.M, d2, self.M)
                constraint = (d2.conj() * MdMT).real.sum(dim=(-2, -1))
            else:
                constraint = torch.einsum(
                    "kn,nm,km->k",
                    delta_flat.conj().to(dtype=self.M.dtype),
                    self.M,
                    delta_flat.to(dtype=self.M.dtype),
                ).real
            mask = constraint > epsilon_dgf**2
            if mask.any():
                scale = epsilon_dgf / torch.sqrt(constraint[mask] + 1e-10)
                delta_flat[mask] = delta_flat[mask] * scale.unsqueeze(1)
            delta = delta_flat.reshape(B, C, H, W)
        else:
            delta = torch.zeros((B, C, H, W), device=self.device, dtype=x.dtype)

        for k in range(self.num_steps):

            delta = delta.detach().requires_grad_(True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(x_tilde + delta)
            loss = self.loss_fn(outputs, y)
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

            grad_flat = grad.reshape(K, n)
            delta_flat = delta.reshape(K, n).detach()

            if is_2d_factorized:
                g2 = grad_flat.reshape(K, H, W).to(self.D_inv_1.dtype)
                DgDT = torch.einsum("ij,kjp,lp->kil", self.D_inv_1, g2, self.D_inv_1)
                dual_sq = (g2.conj() * DgDT).real.sum(dim=(-2, -1))
            else:
                g_c = grad_flat.to(self.D_inv_1.dtype)

                dual_sq = torch.einsum(
                    "kn,nm,km->k", g_c.conj(), self.D_inv_1, g_c
                ).real
            dual_norm = torch.sqrt(dual_sq + 1e-10)

            delta_flat = delta_flat + self.gamma * (grad_flat / dual_norm.unsqueeze(-1))

            with torch.no_grad():
                if is_2d_factorized:
                    delta_flat, _ = project_factorized_2d(
                        delta_flat=delta_flat,
                        U=self.U_M,
                        mu=self.mu_M,
                        H=H,
                        W=W,
                        eps=epsilon_dgf,
                        max_iter=self.proj_max_iter,
                        tol=self.proj_tol,
                    )
                elif self.use_eig:
                    delta_flat, _ = project_CM_fast_eig(
                        delta_flat=delta_flat,
                        U=self.U_M,
                        mu=self.mu_M,
                        eps=epsilon_dgf,
                        max_iter=self.proj_max_iter,
                        tol=self.proj_tol,
                    )
                else:
                    delta_flat, _ = project_CM_batched_solve(
                        delta_flat=delta_flat,
                        Mherm=self.Mherm,
                        eps=epsilon_dgf,
                        max_iter=self.proj_max_iter,
                        tol=self.proj_tol,
                    )

            last_delta_flat = delta_flat
            delta = delta_flat.reshape(B, C, H, W)
            delta = delta.detach().requires_grad_(True)

            if self.verbose and (k + 1) % 5 == 0:
                print(f"  Step {k+1}/{self.num_steps}, Loss: {loss.item():.4f}")

        x_adv = x_tilde + delta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv.detach(), epsilon_dgf, last_delta_flat.detach()

    def attack_case2(
        self, x: torch.Tensor, y: torch.Tensor, random_init: bool = True
    ) -> torch.Tensor:
        ball = "Linf"
        self.gamma = self.epsilon / self.num_steps

        if ball == "L2":
            print("Using Case 2: L2 PGD Attack")
            images = x.clone().detach().to(self.device)
            labels = y.clone().detach().to(self.device)

            adv_images = images.clone().detach()
            batch_size = len(images)

            if random_init:
                delta = torch.empty_like(adv_images).uniform_(
                    -self.epsilon, self.epsilon
                )
            else:
                delta = torch.zeros_like(x)

            def clip_delta_L2(delta, epsilon):
                avoid_zero_div = torch.tensor(
                    1e-12, dtype=delta.dtype, device=delta.device
                )
                reduc_ind = list(range(1, len(delta.size())))

                norm = torch.sqrt(
                    torch.max(
                        avoid_zero_div, torch.sum(delta**2, dim=reduc_ind, keepdim=True)
                    )
                )
                factor = torch.min(
                    torch.tensor(1.0, dtype=delta.dtype, device=delta.device),
                    epsilon / norm,
                )
                delta *= factor

                return delta

            delta = clip_delta_L2(delta, self.epsilon)

            adv_images = torch.clamp(images + delta, 0, 1).detach()

            for _ in range(self.num_steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)

                cost = self.loss_fn(outputs, labels)

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + 1e-10
                grad = grad / grad_norms.view(batch_size, 1, 1, 1)
                adv_images = adv_images.detach() + self.gamma * grad

                delta = adv_images - images
                delta = clip_delta_L2(delta, self.epsilon)

                adv_images = torch.clamp(images + delta, 0, 1)

            return adv_images.detach(), delta.detach()

        elif ball == "Linf":
            print("Using Case 2: Linf PGD Attack")
            delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)

            for _ in range(self.num_steps):
                delta_prime = torch.clamp(delta, 0, 1)
                delta_prime = delta.detach().clone()
                delta_prime.requires_grad = True

                outputs = self.model(x + delta_prime)
                loss = self.loss_fn(outputs, y)

                loss.backward()

                delta_prime = (
                    delta_prime + self.gamma * delta_prime.grad.detach().sign()
                ).clamp(-self.epsilon, self.epsilon)

                delta_prime = torch.min(torch.max(delta_prime, -x), 1 - x)
                delta = delta_prime

            adv_x = torch.clamp(x + delta, 0, 1)
            return adv_x.detach(), delta.detach()

        else:
            raise ValueError("Unsupported ball type. Choose 'L2' or 'Linf'.")

    def attack_case3(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        random_init: bool = True
    ) -> torch.Tensor:
        """
        Case 3: Fourier-based PGD (F-PGD)
        Implements Eq. (7)-(10) from 'A Fourier Perspective of Feature Extraction and Adversarial Robustness' (IJCAI-24).
        """

        def _nyquist_radius(h, w):
            return ( (h//2)**2 + (w//2)**2 ) ** 0.5
        
        def make_radial_mask(
                h, w, band="band", r_low=None, r_high=None,
                r_units="px", 
                device=None, dtype=torch.float32):
            
            assert band in {"low", "high", "band"}
            H, W = int(h), int(w)
            device = device or "cpu"

            yy = torch.arange(-H//2, H - H//2, device=device, dtype=dtype)
            xx = torch.arange(-W//2, W - W//2, device=device, dtype=dtype)
            YY, XX = torch.meshgrid(yy, xx, indexing="ij")
            R = torch.sqrt(YY**2 + XX**2)  # Euclidean radius from the center

            if r_units == "frac":
                nyq = _nyquist_radius(H, W)
                r_low_eff  = None if r_low  is None else float(r_low)  * nyq
                r_high_eff = None if r_high is None else float(r_high) * nyq
            elif r_units == "px":
                r_low_eff, r_high_eff = r_low, r_high
            else:
                raise ValueError("r_units must be 'px' or 'frac'")

            if band == "low":
                assert r_high_eff is not None
                M_centered = (R <= r_high_eff).to(dtype)
            elif band == "high":
                assert r_low_eff is not None
                M_centered = (R >= r_low_eff).to(dtype)
            else:  # band-pass
                assert (r_low_eff is not None) and (r_high_eff is not None)
                M_centered = ((R >= r_low_eff) & (R <= r_high_eff)).to(dtype)

            M_unshifted = torch.fft.ifftshift(M_centered)
            return M_unshifted  # (H, W), real {0,1}

        def apply_freq_mask(tensor, M):
            B, C, H, W = tensor.shape
            M = M.view(1, 1, H, W).to(tensor.device, tensor.dtype)
            F2 = torch.fft.fft2(tensor)       # unshifted FFT
            F2m = F2 * M                      # apply mask
            return torch.fft.ifft2(F2m).real  # back to spatial

        """
        Frequency-constrained PGD with sign + l∞ projection:
        """
        steps = self.num_steps
        eps = self.epsilon
        step_size = eps / steps # self.gamma

        x = x.clone().detach()
        x0 = x.clone()
        B, C, H, W = x.shape

        M = make_radial_mask(H, W, band="low", r_high=0.18, r_units="frac", device=x.device)

        for _ in range(steps):
            x.requires_grad_(True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            grad = torch.autograd.grad(loss, x)[0] 
            x = x.detach()

            g_masked = apply_freq_mask(grad, M)     
            x = x + step_size * g_masked.sign()    

            x = torch.max(torch.min(x, x0 + eps), x0 - eps)

        return x.clamp(0, 1).detach(), (x - x0).detach()

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, random_init: bool = True
    ) -> torch.Tensor:
        self.model.eval()

        if self.case == "case1":
            return self.attack_case1(x, y, random_init)
        elif self.case == "case2":
            return self.attack_case2(x, y, random_init)
        elif self.case == "case3":
            return self.attack_case3(x, y, random_init)
        elif self.case == "case4":
            print("Under construction!")
