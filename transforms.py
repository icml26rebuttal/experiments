from __future__ import annotations
import torch
import numpy as np

'''
The Gabor transform is associated to an overcomplete basis (so-called frame), 
which consists of vectors formed by shifting the so-called window vector in time and frequency. 
The Gabor transform has P=L^2/ab rows, where a and b denote time and frequency parameters, respectively.
'''

def hann_window(l, L):
    return 0.5 * (1 - torch.cos(2 * np.pi * l / (L - 1)))

def blackman_window(l, L):
    return 0.42 - 0.5 * torch.cos(2 * np.pi * l / (L - 1)) + 0.08 * torch.cos(4 * np.pi * l / (L-1))

def gaussian_window(l, L):
    # circularly-centered discrete Gaussian; sigma in samples
    # center at (L-1)/2 for symmetry, then wrap with modulo-L indices already in l
    sigma = 1.0
    c = (L - 1) / 2
    # distance on the circle: min(|l-c|, L-|l-c|)
    d = torch.minimum(torch.remainder(l - c, L), torch.remainder(c - l, L))
    return torch.exp(-(d ** 2) / (2 * sigma ** 2))

def dual_window(Psi, window):
    """
    Based on the frame operator, it computes the dual window of the given window.
    Psi: (M, N), complex
    Returns: (M, N), complex
    """
    S = frameop_DGT(Psi)  # (N, N)
    L = S.shape[0]
    l = torch.arange(L)
    if window == "Hann":
        window_values = hann_window(l, L)
    elif window == "Blackman":
        window_values = blackman_window(l, L)
    elif window == "Gaussian":
        window_values = gaussian_window(l, sigma=1)
    else:
        raise ValueError("No window found.")
    
    window_values = window_values.unsqueeze(-1)  # (N, 1)
    
    S_inv = inv_frameop_DGT(S)  # (N, N)

    Psi_dual = torch.matmul(S_inv, window_values.to(dtype=S_inv.dtype))  # (N, 1)

    return Psi_dual

def DGT(L, a, b, window):
    """
    Generate a digital Gabor transform.

    Args:
        L (int): Dimension of the signal (length of the input signal).
        a (int): Time lattice parameter.
        b (int): Frequency lattice parameter.

    Returns:
        torch.Tensor: Gabor transform matrix of shape (P, L), where P = L^2 / (a * b).
    """
    
    if L % a != 0 or L % b != 0:
        raise ValueError(f"L ({L}) must be divisible by both a ({a}) and b ({b}).")
    
    n_range = torch.arange(int(L/a)) 
    m_range = torch.arange(int(L/b)) 
    l_range = torch.arange(L)

    # Initialize the matrix P x L
    P = int(L/a) * int(L/b)
    gabor_matrix = torch.zeros(P, L, dtype=torch.complex64)

    # Create time-frequency lattice/grid
    n_grid, m_grid, l_grid = torch.meshgrid(n_range, m_range, l_range)

    # Shift the l-grid by n*a and apply modulo L
    shifted_l_grid = (l_grid - n_grid * a) % L

    if window == "Hann":
    # Time-shift the window
        window_values = hann_window(shifted_l_grid, L)
    elif window == "Blackman":
        window_values = blackman_window(shifted_l_grid, L)
    elif window == "Gaussian":
        window_values = gaussian_window(shifted_l_grid, L)
    else:
        raise ValueError("No window found.")

    # Create the frequency-shift for the window
    exp_values = torch.exp(-2j * np.pi * m_grid * b * l_grid / L)

    # Create the digital Gabor transform
    gabor_matrix = exp_values * window_values
    gabor_matrix = gabor_matrix.reshape(gabor_matrix.size(0) * gabor_matrix.size(1), gabor_matrix.size(2))

    return gabor_matrix


def frameop_DGT(Psi):
    """
    Computes the frame operator S = Ψ^* Ψ
    Psi: (M, N), complex
    Returns: (N, N), complex
    """
    Psi_t = Psi.conj().T.to(dtype=Psi.dtype, device=Psi.device)  # (N, M)
    S = torch.matmul(Psi_t, Psi)  # (N, N)
    S_eig = torch.linalg.eigvalsh(S)
    S_halfinv = torch.rsqrt(S_eig) # (N,)

    return S

def inv_frameop_DGT(Psi):
    """
    Computes the inverse of the frame operator S^{-1} = (Ψ^* Ψ)^{-1}
    """
    S = frameop_DGT(Psi)  # (N, N)
    return torch.linalg.inv(S)

def pinv_dgt(Psi, L, a, b, window):
    
    Psi = DGT(L, a, b, window)  # (M, N)
    dual_win = dual_window(Psi, window)  # (N, 1)
    dual_win = dual_win.squeeze(-1)  # (N,)

    n_range = torch.arange(int(L/a)) 
    m_range = torch.arange(int(L/b)) 
    l_range = torch.arange(L)

    # Initialize the matrix P x L
    P = int(L/a) * int(L/b)
    dual_gabor_matrix = torch.zeros(P, L, dtype=torch.complex64)

    # Create time-frequency lattice/grid
    n_grid, m_grid, l_grid = torch.meshgrid(n_range, m_range, l_range)

    # Shift the l-grid by n*a and apply modulo L
    shifted_l_grid = (l_grid - n_grid * a) % L

    dual_window_values = dual_win[shifted_l_grid]  # (n, m, L)

    # Create the frequency-shift for the window
    exp_values = torch.exp(-2j * np.pi * m_grid * b * l_grid / L)

    # Create the digital Gabor transform
    dual_gabor_matrix = exp_values * dual_window_values
    dual_gabor_matrix = dual_gabor_matrix.reshape(dual_gabor_matrix.size(0) * 
                                                  dual_gabor_matrix.size(1), dual_gabor_matrix.size(2))

    psi_pseudoinv = dual_gabor_matrix.conj().T

    return psi_pseudoinv


def diag_frame_potential(
    Psi: torch.Tensor,          # (N,n) complex
    rho: float,
    *,
    num_samples: int = 2048,    # S
    block_i: int = 1024,        # rows processed per block
    eps: float = 1e-6,
    seed: int = 0,
) -> torch.Tensor:
    """
    Approximate FP_i via uniform subsampling of columns j.

    Returns:
        d: (N,) real with d_i = 1/sqrt(FP_i + eps)
    """
    device = Psi.device
    N, _ = Psi.shape

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    S = min(num_samples, N)
    idx = torch.randperm(N, device=device, generator=g)[:S]
    Psi_s = Psi.index_select(0, idx)                      # (S,n)

    FP = torch.empty((N,), device=device, dtype=torch.float32)

    scale = float(N) / float(S)

    for i0 in range(0, N, block_i):
        i1 = min(i0 + block_i, N)
        Psi_i = Psi[i0:i1]                                # (Bi,n)
        G = Psi_i @ Psi_s.conj().T                        # (Bi,S)
        FP_block = (G.abs().pow(2.0 * rho)).sum(dim=1)    # (Bi,)
        FP[i0:i1] = (FP_block.to(torch.float32) * scale)

        del G

    d = 1.0 / torch.sqrt(FP + eps)
    
    return d
################################################################################

# CASE 1 OPERATORS #

################################################################################

"""
Operators needed for case 1:
a) M = Ψ* D Ψ -----> weights1
b) Projection operator: P = (I + 2λ M)^(-1) -----> projection_op1
c) Dual norm: Ψ* D^(-1) Ψ -----> dual_norm1
"""

def weights1(D, Psi):

    # DPsi = torch.matmul(D, Psi) # for full diagonal matrix D
    DPsi = D.unsqueeze(-1) * Psi
    M = torch.matmul(Psi.conj().T, DPsi)

    return M

def projection_op1(D, Psi, lamda):

    M = weights1(D, Psi)

    idn = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)

    proj = (idn + 2 * lamda * M)
    inv_proj = torch.linalg.inv(proj)

    return inv_proj

def dual_norm1(D, Psi):

    Dinv = D.pow_(-1)
    DinvPsi = Dinv.unsqueeze(-1) * Psi
    Psit_DinvPsi = torch.matmul(Psi.conj().T, DinvPsi)

    return Psit_DinvPsi


def diag_weights_from_mc_row_sums(
    Psi,
    num_j=4000,
    eps_norm=1e-12,
    eps_weight=1e-3,
    p=1.0,
    mode="down",         # "down" or "up"
    batch_rows=8192,
    seed=None,
):
    """
    Build diagonal weights d (length N) for D=diag(d) using MC approximation of
    c_i = |sum_j <psi_i, psi_j>| with normalized rows psi_i.

    Returns: d (float32, shape N)
    """
    A = Psi.to(torch.complex64) if not torch.is_complex(Psi) else Psi
    A = A / A.norm(dim=1, keepdim=True).clamp_min(eps_norm)  # normalize rows

    N, n = A.shape

    g = None
    if seed is not None:
        g = torch.Generator(device=A.device)
        g.manual_seed(seed)

    m = min(num_j, N)
    J = torch.randint(0, N, (m,), device=A.device, generator=g)

    u_hat = (N / m) * A[J].sum(dim=0)  # (n,)

    # compute c_i_hat = |<psi_i, u_hat>| for all i, in batches
    c = torch.empty(N, device=A.device, dtype=torch.float32)
    for i0 in range(0, N, batch_rows):
        i1 = min(i0 + batch_rows, N)
        s_blk = (A[i0:i1] * u_hat.conj()).sum(dim=1)  # <psi_i, u_hat>
        c[i0:i1] = s_blk.abs().real.to(torch.float32)

    # scale like paper's (N-1) factor if you want that normalization
    c = c / (N - 1)

    if mode == "down":
        print("Using 'down' mode for diag weights.")
        d = 1.0 / (eps_weight + c).pow(p)
    elif mode == "up":
        print("Using 'up' mode for diag weights.")
        d = (eps_weight + c).pow(p)
    else:
        raise ValueError("mode must be 'down' or 'up'")

    return d

################################################################################

# SOFT THRESHOLDING #

################################################################################

def soft_thresholding(x, threshold):
    """
    Apply soft-thresholding to the input tensor x with the given threshold.

    Args:
        x (torch.Tensor): Input tensor.
        threshold (float): Threshold value.

    Returns:
        torch.Tensor: Tensor after applying soft-thresholding.
    """
    
    if torch.is_complex(x):
        magnitude = torch.abs(x)
        phase = x / (magnitude + 1e-8)  # Avoid division by zero

        magnitude_thresholded = torch.where(magnitude >= threshold, phase * (magnitude - threshold), torch.zeros_like(magnitude))
        return magnitude_thresholded
    else:
        return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))



"""
Projection onto the ellipsoid C_M = {x : x^T M x <= eps^2} for symmetric PD M.

We split into two functions:
  1) _lambda_bisection_from_eigcoords(...): finds the unique lambda >= 0 (0 if already feasible), 
  using the bisection method
  2) project_CM_fast(...): computes x = (I + 2 lambda M)^(-1) delta

Implementation uses an eigendecomposition of M for stability & speed:
  M = U diag(mu) U^T, mu >= 0.
Then:
  x(lambda) = U diag(1/(1+2 lambda mu)) U^T delta
and
  x(lambda)^T M x(lambda) = sum_i mu_i z_i^2 / (1+2 lambda mu_i)^2, where z = U^T delta.
"""

def _lambda_bisection_from_eigcoords(
    z: torch.Tensor,         # (K,n) complex, z = delta @ U
    mu: torch.Tensor,        # (n,) real >=0
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Solve per-row lambda for:
        q(lambda) = sum_i mu_i |z_i|^2 / (1+2 lambda mu_i)^2 = eps^2
    Returns lam (K,) float32 with lam=0 for feasible rows.
    """
    device = z.device
    K, n = z.shape
    eps2 = eps * eps

    mu = mu.to(device).clamp_min(0.0)
    w = z.abs().square()                                 # (K,n)
    q0 = (w * mu.unsqueeze(0)).sum(dim=-1)               # (K,)

    inside = q0 <= eps2
    lam = torch.zeros((K,), device=device, dtype=torch.float32)
    if inside.all():
        return lam

    idx = (~inside).nonzero(as_tuple=False).squeeze(-1)
    w_sub = w.index_select(0, idx)                       # (Kout,n)
    Kout = w_sub.shape[0]
    mu_row = mu.unsqueeze(0)                             # (1,n)

    def q_of(lam_vec):
        denom2 = (1.0 + 2.0 * lam_vec.unsqueeze(-1) * mu_row).square()
        return (mu_row * w_sub / denom2).sum(dim=-1)

    lo = torch.zeros((Kout,), device=device, dtype=torch.float32)
    hi = torch.ones((Kout,), device=device, dtype=torch.float32)

    q_hi = q_of(hi)
    for _ in range(40):
        need = q_hi > eps2
        if not need.any():
            break
        hi[need] *= 2.0
        q_hi = q_of(hi)
    else:
        raise RuntimeError("Failed to bracket lambda. Check M PSD and eps > 0.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        q_mid = q_of(mid)
        right = q_mid <= eps2
        hi = torch.where(right, mid, hi)
        lo = torch.where(right, lo, mid)
        if (q_mid - eps2).abs().max().item() < tol:
            break

    lam_sub = hi
    lam.index_copy_(0, idx, lam_sub)
    return lam


def project_CM_fast_eig(
    delta_flat: torch.Tensor,   # (K,n) real
    U: torch.Tensor,            # (n,n) complex unitary eigenvectors of M
    mu: torch.Tensor,           # (n,) real eigenvalues of M
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fast projection using cached eigendecomposition M = U diag(mu) U^H.
    Returns projected delta (K,n) real and lambdas (K,).
    """
    device = delta_flat.device
    delta_c = delta_flat.to(U.dtype)  # complex work dtype

    # z = U^H delta (row-batched convention)
    z = torch.matmul(delta_c, U)  # (K,n) complex

    lam = _lambda_bisection_from_eigcoords(z, mu, eps, max_iter=max_iter, tol=tol)

    inv = 1.0 / (1.0 + 2.0 * lam.unsqueeze(-1) * mu.to(device).unsqueeze(0))  # (K,n) real
    proj_c = torch.matmul(z * inv, U.conj().T) # (K,n) complex

    return proj_c.real, lam



def project_CM_batched_solve(
    delta_flat: torch.Tensor,   # (K,n) real
    Mherm: torch.Tensor,         # (n,n) complex/real Hermitian PSD-ish
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
    bracket_iters: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Robust projection using bisection on lambda and batched linear solves:
        x(lambda) = (I + 2 lambda M)^(-1) delta

    Returns:
        proj_flat: (K,n) real
        lam:       (K,) float32
    """
    device = delta_flat.device
    K, n = delta_flat.shape
    eps2 = eps * eps

    dtype = Mherm.dtype
    delta_c = delta_flat.to(dtype)

    M = 0.5 * (Mherm + Mherm.mH)

    q0 = torch.einsum('kn,nm,km->k', delta_c.conj(), M, delta_c).real
    inside = q0 <= eps2

    lam = torch.zeros((K,), device=device, dtype=torch.float32)
    proj = delta_c.clone()

    if inside.all():
        return delta_flat, lam

    idx = (~inside).nonzero(as_tuple=False).squeeze(-1)
    d = delta_c.index_select(0, idx)  # (Kout,n)
    Kout = d.shape[0]

    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)  # (1,n,n)
    Mbat = M.unsqueeze(0)                                       # (1,n,n)

    def solve_and_q(lam_vec: torch.Tensor):
        lamv = lam_vec.to(dtype=dtype)
        A = I + (2.0 * lamv).view(Kout, 1, 1) * Mbat
        x = torch.linalg.solve(A, d.unsqueeze(-1)).squeeze(-1)
        q = torch.einsum('kn,nm,km->k', x.conj(), M, x).real
        return x, q

    lo = torch.zeros((Kout,), device=device, dtype=torch.float32)
    hi = torch.ones((Kout,), device=device, dtype=torch.float32)

    _, q_hi = solve_and_q(hi)
    for _ in range(bracket_iters):
        need = q_hi > eps2
        if not need.any():
            break
        hi[need] *= 2.0
        _, q_hi = solve_and_q(hi)
    else:
        raise RuntimeError("Failed to bracket lambda. Check eps > 0 and M PSD-ish.")

    x_mid = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x_mid, q_mid = solve_and_q(mid)
        right = q_mid <= eps2
        hi = torch.where(right, mid, hi)
        lo = torch.where(right, lo, mid)
        if (q_mid - eps2).abs().max().item() < tol:
            break

    proj.index_copy_(0, idx, x_mid)
    lam.index_copy_(0, idx, hi)

    return proj.real, lam