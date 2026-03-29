"""
DGF-PGD Attack Implementation (Based on Algorithm 1)
====================================================

- Case 1: DGF-PGD attack
- Case 2: Standard linf PGD attack
- Case 3: Fourier-based linf PGD attack
"""

import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.fft
import torch.nn as nn
from typing import Callable, Optional, Tuple
from transforms import *

class DGFPGDAttack:
    """
    Implements Case 1
    Models always receive (B, C, H, W) formatted inputs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        Psi_2D: torch.Tensor,  # 2D Analysis operator (N_2D x n_2D) complex
        Psi_plus_2D: torch.Tensor,  # 2D Pseudo-inverse (n_2D x N_2D) complex
        D_inv_1: torch.Tensor,  # Inverse of D (N x N)
        M: Optional[torch.Tensor],  # Case 1: Ψ*DΨ (n x n) complex
        eps_scale: Optional[float],  # Scaling for epsilon in DGF-PGD
        mu_M: Optional[torch.Tensor],  # Eigenvalues of M (n,)
        U_M: Optional[torch.Tensor],  # Eigenvectors of M (n,n)
        M_herm: Optional[torch.Tensor],  # Hermitian of M (n,n)
        use_eig: Optional[bool],  # Whether to
        image_shape: Tuple[int, int, int] = (3, 64, 64),  # (C, H, W)
        tau: float = 0.1,  # Soft-thresholding parameter
        rho : float = 1.0,  # Frame potential parameter
        epsilon: float = 0.1,  # Attack level ε of spatial attacks; it is updated accordingly for DGF-PGD
        gamma: float = 0.01,  # Step size γ (called alpha in paper sometimes)
        num_steps: int = 10,  # Number of iterations K
        case = 'case1',  # 'case1' for Case 1, 'case2' for Case 2, 'case3' for Case 3
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = False
    ):
        """
        Initialize DGF-PGD attack
        
        Args:
            model: Target model (expects (B, C, H, W))
            loss_fn: Loss function
            Psi: Analysis operator (N x n, complex)
            Psi_plus: Pseudo-inverse (n x N, complex)
            D: Diagonal weight matrix (N x N, real)
            D_inv: (Ψ^* D^-1 Ψ) (N x N, complex)
            M: Case 1 weight matrix (n x n, complex)
            image_shape: (C, H, W)
            tau: Soft-thresholding parameter
            epsilon: Attack level ε for spatial attacks, e.g., l2-PGD
            gamma: Step size γ for the DGF-PGD updates
            num_steps: Number of PGD iterations K
            case1: If True, use Case 1 attack; if False, use another Case attack
            device: Device
            verbose: Print progress
        """
        self.model = model
        self.loss_fn = loss_fn
        self.tau = tau
        self.rho = rho
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_steps = num_steps
        self.case = case
        self.device = device
        self.verbose = verbose

         # Move operators to device
        self.Psi_2D = Psi_2D.to(device)
        self.Psi_plus_2D = Psi_plus_2D.to(device)

        self.M = M.to(device)  # M = Ψ^* D Ψ
        self.D_inv_1 = D_inv_1.to(device) # D_inv_1 = Ψ^* D^{-1} Ψ
        
        # Image parameters
        self.C, self.H, self.W = image_shape
        self.n = self.H * self.W
        self.N = self.Psi_2D.shape[0]

        # Projection settings
        self.proj_max_iter = 25
        self.proj_tol = 1e-5
        # self.jitter_scale = 1e-6

        self.eps_scale = eps_scale
        self.mu_M = mu_M.to(device)
        self.U_M = U_M.to(device)
        self.Mherm = M_herm.to(device)
        self.use_eig = use_eig
        
        if self.verbose:
            print("DGF-PGD Attack (Fast, Case 1 per-channel)")
            print(f"  Image shape: ({self.C},{self.H},{self.W}), n={self.n}, N={self.N}")
            print(f"  eps={self.epsilon}, gamma={self.gamma}, steps={self.num_steps}")
            print(f"  projection: {'eig' if self.use_eig else 'solve-fallback'}")
            print(f"  eps_scale={self.eps_scale:.6g},\
                   eps_dgf={(self.eps_scale * self.epsilon):.6g}")
    
    def soft_threshold(self, z: torch.Tensor) -> torch.Tensor:
        """
        Complex soft-thresholding S_τ (see Notation section)
        
        S_τ(x) = (x/|x|)(|x| - τ) if |x| ≥ τ, else 0
        """
        magnitude = torch.abs(z)
        mask = magnitude >= self.tau
        
        result = torch.zeros_like(z)
        result[mask] = (z[mask] / magnitude[mask]) * (magnitude[mask] - self.tau)
        
        return result
    
    def attack_case1(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        random_init: bool = True
    ) -> torch.Tensor:
        """
        Case 1: Soft-thresholded frame attack (Algorithm 1)
        
        Algorithm steps:
        1. Given a standard attack level epsilon, compute ε_DGF = exp( log|det(M)| / (H*W) )
        2. Initialize δ_0 ~ U(-ζε_DGF, ζε_DGF) with ζ = ||Ψ^+||_op
        3. Compute x̃ = Ψ^+ S_τ(Ψx)
        4. For k = 0 to K-1:
             g̃_k = ∇L(x̃ + δ^k)
             δ^{k+1} = Π_C(δ^k + γ g̃_k / sqrt((g̃_k)^T D^{-1} g̃_k))
        5. x_adv = x̃ + δ^K

        Args:
            x: Clean images (B, C, H, W)
            y: Labels (B,)
            random_init: Random initialization
            
        Returns:
            Adversarial images (B, C, H, W)
        """
        B, C, H, W = x.shape
        if (C, H, W) != (self.C, self.H, self.W):
            raise ValueError(f"Input shape mismatch: got ({C},{H},{W}) expected ({self.C},{self.H},{self.W})")

        K = B * C
        n = self.n

        epsilon_dgf = self.eps_scale * self.epsilon
        # self.gamma = epsilon_dgf / self.num_steps  # Step size

        # ---- Step 1: x_tilde = Psi^+ (S_tau(Psi x Psi^T)) Psi^+----
        n1 = self.Psi_2D.shape[1]
        Psi_bc = self.Psi_2D.view(1, 1, self.N, n1)  
        PsiT_bc = self.Psi_2D.t().view(1, 1, n1, self.N)
        z = torch.matmul(Psi_bc, x.to(torch.complex64))
        z = torch.matmul(z, PsiT_bc)
        z = soft_thresholding(z, self.tau)

        Psi_plus_bc = self.Psi_plus_2D.view(1, 1, n1, self.N)
        Psi_plusT_bc = self.Psi_plus_2D.t().view(1, 1, self.N, n1)
        x_tilde = torch.matmul(Psi_plus_bc, z)
        x_tilde = torch.matmul(x_tilde, Psi_plusT_bc).real
        x_tilde = torch.clamp(x_tilde, 0.0, 1.0)
        
        # ---- Step 2: init delta ----
        if random_init:
            delta = torch.empty((B, C, H, W), device=self.device, dtype=x.dtype)
            # delta.uniform_(-self.zeta * epsilon_dgf, self.zeta * epsilon_dgf)
            delta.uniform_(-epsilon_dgf, epsilon_dgf)
            delta_flat = delta.reshape(K, n)
            constraint = torch.einsum("kn,nm,km->k", delta_flat.conj().to(dtype=self.M.dtype), 
                                      self.M, delta_flat.to(dtype=self.M.dtype)).real  # (K,)
            mask = constraint > epsilon_dgf ** 2
            if mask.any():
                scale = epsilon_dgf / torch.sqrt(constraint[mask] + 1e-10)
                delta_flat[mask] = delta_flat[mask] * scale.unsqueeze(1)
            delta = delta_flat.reshape(B, C, H, W)
        else:
            delta = torch.zeros((B, C, H, W), device=self.device, dtype=x.dtype)
        delta = torch.zeros((B, C, H, W), device=self.device, dtype=x.dtype)
        
        # ---- Step 3: PGD iterations ----
        for k in range(self.num_steps):

            delta = delta.detach().requires_grad_(True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(x_tilde + delta)
            loss = self.loss_fn(outputs, y)
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]  # (B,C,H,W)

            grad_flat = grad.reshape(K, n)                 # (K,n)
            delta_flat = delta.reshape(K, n).detach()      # (K,n)

            # dual norm: sqrt(g^H D_inv_1 g) per row
            g_c = grad_flat.to(self.D_inv_1.dtype)
            
            dual_sq = torch.einsum('kn,nm,km->k', g_c.conj(), self.D_inv_1, g_c).real
            dual_norm = torch.sqrt(dual_sq + 1e-10)        # (K,)

            # normalized step
            delta_flat = delta_flat + self.gamma * (grad_flat / dual_norm.unsqueeze(-1))
            
            # projection (vectorized)
            with torch.no_grad():
                if self.use_eig:
                    delta_flat, _lam = project_CM_fast_eig(
                        delta_flat=delta_flat,
                        U=self.U_M,
                        mu=self.mu_M,
                        eps=epsilon_dgf,
                        max_iter=self.proj_max_iter,
                        tol=self.proj_tol,
                    )
                else:
                    delta_flat, _lam = project_CM_batched_solve(
                        delta_flat=delta_flat,
                        Mherm=self.Mherm,
                        eps=epsilon_dgf,
                        max_iter=self.proj_max_iter,
                        tol=self.proj_tol,
                    )


            # keep the output of DGF-PGD, i.e. the implemented attack, to be passed for evaluation
            last_delta_flat = delta_flat
            delta = delta_flat.reshape(B, C, H, W)
            delta = delta.detach().requires_grad_(True)

            if self.verbose and (k + 1) % 5 == 0:
                print(f"  Step {k+1}/{self.num_steps}, Loss: {loss.item():.4f}")
                
        # ---- Step 4: final adversarial example ----
        # pass epsilon_dgf for evaluation purposes
        x_adv = x_tilde + delta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        return x_adv.detach(), epsilon_dgf, last_delta_flat.detach()
        
    
    def attack_case2(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        random_init: bool = True
    ) -> torch.Tensor:
        
        """
        Case 2: standard PGD attack
        Algorithm steps:
        1. Initialize δ_0 ~ U(-ε, ε)
        2. For k = 0 to K-1:
             g_k = ∇L(x + δ^k)
             δ^{k+1} = δ^k + γ g_k / ||g_k||_2
        3. x_adv = x + δ^K
        Args:
            x: Clean images (B, C, H, W)
            y: Labels (B,)
            random_init: Random initialization
        """

        # Case 2: PGD attack 

        ball = 'Linf'  # Choose between 'Linf' or 'L2'
        self.gamma = self.epsilon / self.num_steps  # Step size
        
        ###########
        # with projection in L2 ball
        ###########

        if ball == 'L2':
            print("Using Case 2: L2 PGD Attack")
            images = x.clone().detach().to(self.device)
            labels = y.clone().detach().to(self.device)

            adv_images = images.clone().detach()
            batch_size = len(images)

            if random_init:
                # Starting at a uniformly random point
                delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            else:
                delta = torch.zeros_like(x)

            def clip_delta_L2(delta, epsilon):
                avoid_zero_div = torch.tensor(1e-12, dtype=delta.dtype, device=delta.device)
                reduc_ind = list(range(1, len(delta.size())))

                norm = torch.sqrt(
                    torch.max(
                        avoid_zero_div, torch.sum(delta ** 2, dim=reduc_ind, keepdim=True)
                    )
                )
                factor = torch.min(
                    torch.tensor(1.0, dtype=delta.dtype, device=delta.device), epsilon / norm
                )
                delta *= factor

                return delta
            
            delta = clip_delta_L2(delta, self.epsilon)

            adv_images = torch.clamp(images + delta, 0, 1).detach()

            for _ in range(self.num_steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)

                # Calculate loss
                cost = self.loss_fn(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                grad_norms = (
                    torch.norm(grad.view(batch_size, -1), p=2, dim=1)
                    + 1e-10
                )  # nopep8
                grad = grad / grad_norms.view(batch_size, 1, 1, 1)
                adv_images = adv_images.detach() + self.gamma * grad

                delta = adv_images - images
                delta = clip_delta_L2(delta, self.epsilon)

                adv_images = torch.clamp(images + delta, 0, 1)

            return adv_images.detach(), delta.detach()
        
        ###########
        # with projection in L_inf ball
        ###########

        elif ball == 'Linf':
            print("Using Case 2: Linf PGD Attack")
        # Random initialization
            delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)

            for _ in range(self.num_steps):
                delta_prime = torch.clamp(delta, 0, 1)
                delta_prime = delta.detach().clone()
                delta_prime.requires_grad = True

                outputs = self.model(x + delta_prime)
                loss = self.loss_fn(outputs, y)    

                loss.backward()

                # PGD step
                delta_prime = (
                    delta_prime + self.gamma * delta_prime.grad.detach().sign()
                ).clamp(-self.epsilon, self.epsilon)

                # Projection step
                # keep pixel values in [0,1]
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

        Steps:
        1. Compute gradient δ_t = ∇_x L(x_t, y; θ)
        2. Apply FFT and frequency mask: δ_t^f = IFFT(FFT(δ_t) ⊙ M)
        3. Update using relaxed F-PGD-L rule:
            x_{t+1} = x_t + α * card(g_f) / ||g_f||_1 * g_f
        4. Project perturbations back into L2 ball of radius ε
        5. Clamp to [0, 1]

        Args:
            x: Clean images (B, C, H, W)
            y: Labels (B,)
            random_init: Random initialization
        """

        def _nyquist_radius(h, w):
            # Euclidean radius from DC (center) to the furthest frequency bin (corner) in a centered FFT.
            return ( (h//2)**2 + (w//2)**2 ) ** 0.5
        
        def make_radial_mask(
                h, w, band="band", r_low=None, r_high=None,
                r_units="px",  # "px" for pixel-index radii, "frac" for fraction of Nyquist radius
                device=None, dtype=torch.float32):
            """
            Build a centered radial pass mask M_centered, then ifftshift to match torch.fft.fft2 layout.

            band:   "low"  -> pass |f| <= r_high
                    "high" -> pass |f| >= r_low
                    "band" -> pass r_low <= |f| <= r_high

            r_units:
                - "px"   : r given in index units on a centered grid (0..~22.6 for 32x32)
                - "frac" : r given as a fraction of Nyquist radius in [0,1], dataset-agnostic

            Returns:
                M (H, W), real {0,1}, already in **unshifted** layout to multiply with fft2 output.

            Notes for CIFAR-10/100 (H=W=32):
                Nyquist radius ≈ sqrt(16^2 + 16^2) ≈ 22.627.
                Examples:
                - low-pass up to r=4 px: r_high=4, r_units="px"
                - mid-band 6..10 px: r_low=6, r_high=10
                - high-pass >= 12 px: r_low=12
                As fractions: 4/22.627≈0.177, 10/22.627≈0.442, etc.
            """
            assert band in {"low", "high", "band"}
            H, W = int(h), int(w)
            device = device or "cpu"

            # Centered coordinate grid: [-floor(H/2), ..., +ceil(H/2)-1]
            yy = torch.arange(-H//2, H - H//2, device=device, dtype=dtype)
            xx = torch.arange(-W//2, W - W//2, device=device, dtype=dtype)
            YY, XX = torch.meshgrid(yy, xx, indexing="ij")
            R = torch.sqrt(YY**2 + XX**2)  # Euclidean radius from the center

            # Convert radii if given as a fraction of Nyquist
            if r_units == "frac":
                nyq = _nyquist_radius(H, W)
                r_low_eff  = None if r_low  is None else float(r_low)  * nyq
                r_high_eff = None if r_high is None else float(r_high) * nyq
            elif r_units == "px":
                r_low_eff, r_high_eff = r_low, r_high
            else:
                raise ValueError("r_units must be 'px' or 'frac'")

            # Build centered mask
            if band == "low":
                assert r_high_eff is not None
                M_centered = (R <= r_high_eff).to(dtype)
            elif band == "high":
                assert r_low_eff is not None
                M_centered = (R >= r_low_eff).to(dtype)
            else:  # band-pass
                assert (r_low_eff is not None) and (r_high_eff is not None)
                M_centered = ((R >= r_low_eff) & (R <= r_high_eff)).to(dtype)

            # Convert to unshifted layout so it aligns with torch.fft.fft2 output
            M_unshifted = torch.fft.ifftshift(M_centered)
            return M_unshifted  # (H, W), real {0,1}

        def apply_freq_mask(tensor, M):
            """
            tensor: (B,C,H,W) real
            M     : (H,W) mask in **unshifted** FFT layout (output of make_radial_mask)
            Returns: masked gradient back in spatial domain, real (B,C,H,W).

            Implements δ^f = IFFT( FFT(δ) ⊙ M ). For real-valued inputs, using a symmetric
            radial mask preserves Hermitian structure (and the IFFT result is real up to fp error).
            """
            B, C, H, W = tensor.shape
            M = M.view(1, 1, H, W).to(tensor.device, tensor.dtype)
            F2 = torch.fft.fft2(tensor)       # unshifted FFT
            F2m = F2 * M                      # apply mask
            return torch.fft.ifft2(F2m).real  # back to spatial

        """
        Frequency-constrained PGD with sign + l∞ projection:

        (7) grad_t     = ∇_x L(x_t, y)
        (8) grad_t^f   = IFFT( FFT(grad_t) ⊙ M )
        (9) x_{t+1}    = clip_{x,ε}( x_t + α * sign(grad_t^f) )

        Args:
        x: (B,C,H,W) input (same space the model expects; if normalized, eps/α must match)
        y: (B,) labels
        eps: l∞ budget around x_orig
        step_size (α): defaults to eps/steps if None
        band/r_low/r_high: frequency pass region (see make_radial_mask)
        targeted: use -loss for targeted attack
        x_min/x_max: valid input bounds in the same space as x
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
            grad = torch.autograd.grad(loss, x)[0]  # Eq. (7)
            x = x.detach()

            g_masked = apply_freq_mask(grad, M)     # Eq. (8)
            x = x + step_size * g_masked.sign()     # Eq. (9) sign step

            # project to l∞(x0, eps) then clamp to data range
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)

        return x.clamp(0, 1).detach(), (x - x0).detach()
        # images = x.clone().detach().to(self.device)
        # labels = y.clone().detach().to(self.device)
        # adv_images = images.clone().detach()
        # batch_size = len(images)
        # eps = 1e-12
        # print("Using Case 3: Fourier-based PGD Attack")

        # # Random L2 initialization
        # # ------------------------------------------------------------------
        # if random_init:
        #         # Starting at a uniformly random point
        #         delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        #         # d_flat = delta.view(adv_images.size(0), -1)
        #         # n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
        #         # r = torch.zeros_like(n).uniform_(0, 1)
        #         # delta *= r / n * self.epsilon
        #         adv_images = torch.clamp(images + delta, 0, 1).detach()
        # else:
        #         delta = torch.zeros_like(x)
        #         adv_images = images.detach()

        # # ------------------------------------------------------------------
        # # Iterative Fourier-based PGD
        # # ------------------------------------------------------------------
        # for _ in range(self.num_steps):
        #     adv_images.requires_grad = True
        #     outputs = self.model(adv_images)
        #     loss = self.loss_fn(outputs, labels)

        #     # Compute gradient
        #     grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        #     # --------------------------------------------------------------
        #     # Fourier step with random binary mask (Eq. 7–8)
        #     # --------------------------------------------------------------
        #     grad_fft = torch.fft.fft2(grad, norm="ortho")
        #     grad_fft_shift = torch.fft.fftshift(grad_fft)

        #     # Create a simple random binary mask (new each iteration)
        #     H, W = grad_fft_shift.shape[-2:]
        #     mask = (torch.rand((H, W), device=self.device) > 0.5).float()
        #     grad_fft_masked = grad_fft_shift * mask[None, None, :, :]

        #     grad_fft_masked = torch.fft.ifftshift(grad_fft_masked)
        #     grad_filtered = torch.fft.ifft2(grad_fft_masked, norm="ortho").real

        #     # --------------------------------------------------------------
        #     # Relaxed update (Eq. 10)
        #     # --------------------------------------------------------------
        #     grad_flat = grad_filtered.view(batch_size, -1)
        #     norm1 = grad_flat.norm(p=1, dim=1, keepdim=True)
        #     card = torch.tensor(grad_flat.size(1), device=x.device, dtype=grad_flat.dtype)
        #     step = (self.gamma * card / (norm1 + eps)).view(-1, 1, 1, 1) * grad_filtered

        #     adv_images = adv_images.detach() + step

        #     # --------------------------------------------------------------
        #     # Projection to L2 ball
        #     # --------------------------------------------------------------
        #     delta = adv_images - images
        #     delta_flat = delta.view(batch_size, -1)
        #     delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True)
        #     factor = torch.min(torch.ones_like(delta_norm), self.epsilon / (delta_norm + eps))
        #     delta = (delta_flat * factor).view_as(delta)
        #     adv_images = torch.clamp(images + delta, 0, 1).detach()

        # return adv_images.detach(), delta.detach()

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        random_init: bool = True
    ) -> torch.Tensor:
        """
        Execute DGF-PGD attack
        
        Args:
            x: Clean images (B, C, H, W)
            y: Labels (B,)
            case: 1 or 2
            random_init: Random initialization
            
        Returns:
            Adversarial images (B, C, H, W)
        """
        self.model.eval()
        
        if self.case == 'case1':
            return self.attack_case1(x, y, random_init)
        elif self.case == 'case2':
            return self.attack_case2(x, y, random_init)
        elif self.case == 'case3':
            return self.attack_case3(x, y, random_init)