from __future__ import annotations
import torch
import numpy as np


def hann_window(l, L):
    return 0.5 * (1 - torch.cos(2 * np.pi * l / (L - 1)))


def blackman_window(l, L):
    return (
        0.42
        - 0.5 * torch.cos(2 * np.pi * l / (L - 1))
        + 0.08 * torch.cos(4 * np.pi * l / (L - 1))
    )


def gaussian_window(l, L):
    sigma = 1.0
    c = (L - 1) / 2
    d = torch.minimum(torch.remainder(l - c, L), torch.remainder(c - l, L))
    return torch.exp(-(d**2) / (2 * sigma**2))


def dual_window(Psi, window):
    S = frameop_DGT(Psi)
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

    window_values = window_values.unsqueeze(-1)

    S_inv = inv_frameop_DGT(S)

    Psi_dual = torch.matmul(S_inv, window_values.to(dtype=S_inv.dtype))

    return Psi_dual


def DGT(L, a, b, window):
    if L % a != 0 or L % b != 0:
        raise ValueError(f"L ({L}) must be divisible by both a ({a}) and b ({b}).")

    n_range = torch.arange(int(L / a))
    m_range = torch.arange(int(L / b))
    l_range = torch.arange(L)

    P = int(L / a) * int(L / b)
    gabor_matrix = torch.zeros(P, L, dtype=torch.complex64)

    n_grid, m_grid, l_grid = torch.meshgrid(n_range, m_range, l_range)

    shifted_l_grid = (l_grid - n_grid * a) % L

    if window == "Hann":
        window_values = hann_window(shifted_l_grid, L)
    elif window == "Blackman":
        window_values = blackman_window(shifted_l_grid, L)
    elif window == "Gaussian":
        window_values = gaussian_window(shifted_l_grid, L)
    else:
        raise ValueError("No window found.")

    exp_values = torch.exp(-2j * np.pi * m_grid * b * l_grid / L)

    gabor_matrix = exp_values * window_values
    gabor_matrix = gabor_matrix.reshape(
        gabor_matrix.size(0) * gabor_matrix.size(1), gabor_matrix.size(2)
    )

    return gabor_matrix


def frameop_DGT(Psi):
    Psi_t = Psi.conj().T.to(dtype=Psi.dtype, device=Psi.device)
    S = torch.matmul(Psi_t, Psi)
    return S


def inv_frameop_DGT(Psi):
    S = frameop_DGT(Psi)
    return torch.linalg.inv(S)


def pinv_dgt(Psi, L, a, b, window):

    Psi = DGT(L, a, b, window)
    dual_win = dual_window(Psi, window)
    dual_win = dual_win.squeeze(-1)

    n_range = torch.arange(int(L / a))
    m_range = torch.arange(int(L / b))
    l_range = torch.arange(L)

    P = int(L / a) * int(L / b)
    dual_gabor_matrix = torch.zeros(P, L, dtype=torch.complex64)

    n_grid, m_grid, l_grid = torch.meshgrid(n_range, m_range, l_range)

    shifted_l_grid = (l_grid - n_grid * a) % L

    dual_window_values = dual_win[shifted_l_grid]

    exp_values = torch.exp(-2j * np.pi * m_grid * b * l_grid / L)

    dual_gabor_matrix = exp_values * dual_window_values
    dual_gabor_matrix = dual_gabor_matrix.reshape(
        dual_gabor_matrix.size(0) * dual_gabor_matrix.size(1), dual_gabor_matrix.size(2)
    )

    psi_pseudoinv = dual_gabor_matrix.conj().T

    return psi_pseudoinv


def diag_frame_potential(
    Psi: torch.Tensor,
    rho: float,
    *,
    num_samples: int = 2048,
    block_i: int = 1024,
    eps: float = 1e-6,
    seed: int = 0,
) -> torch.Tensor:
    device = Psi.device
    N, _ = Psi.shape

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    S = min(num_samples, N)
    idx = torch.randperm(N, device=device, generator=g)[:S]
    Psi_s = Psi.index_select(0, idx)

    FP = torch.empty((N,), device=device, dtype=torch.float32)

    scale = float(N) / float(S)

    for i0 in range(0, N, block_i):
        i1 = min(i0 + block_i, N)
        Psi_i = Psi[i0:i1]
        G = Psi_i @ Psi_s.conj().T
        FP_block = (G.abs().pow(2.0 * rho)).sum(dim=1)
        FP[i0:i1] = FP_block.to(torch.float32) * scale

        del G

    d = 1.0 / torch.sqrt(FP + eps)

    return d


def weights1(D, Psi):

    DPsi = D.unsqueeze(-1) * Psi
    M = torch.matmul(Psi.conj().T, DPsi)

    return M


def projection_op1(D, Psi, lamda):

    M = weights1(D, Psi)

    idn = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)

    proj = idn + 2 * lamda * M
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
    mode="down",
    batch_rows=8192,
    seed=None,
):
    A = Psi.to(torch.complex64) if not torch.is_complex(Psi) else Psi
    A = A / A.norm(dim=1, keepdim=True).clamp_min(eps_norm)

    N, n = A.shape

    g = None
    if seed is not None:
        g = torch.Generator(device=A.device)
        g.manual_seed(seed)

    m = min(num_j, N)
    J = torch.randint(0, N, (m,), device=A.device, generator=g)

    u_hat = (N / m) * A[J].sum(dim=0)

    c = torch.empty(N, device=A.device, dtype=torch.float32)
    for i0 in range(0, N, batch_rows):
        i1 = min(i0 + batch_rows, N)
        s_blk = (A[i0:i1] * u_hat.conj()).sum(dim=1)
        c[i0:i1] = s_blk.abs().real.to(torch.float32)

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


def distord(x, threshold):
    if torch.is_complex(x):
        spectral_amplitude = torch.abs(x)
        polar_unit_phasor = x / (spectral_amplitude + 1e-8)

        shrunk_complex_coefficient = torch.where(
            spectral_amplitude >= threshold,
            polar_unit_phasor * (spectral_amplitude - threshold),
            torch.zeros_like(spectral_amplitude),
        )
        return shrunk_complex_coefficient
    else:
        return torch.sign(x) * torch.max(
            torch.abs(x) - threshold, torch.tensor(0.0, device=x.device)
        )


def _lambda_bisection_from_eigcoords(
    z: torch.Tensor,
    mu: torch.Tensor,
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> torch.Tensor:
    device = z.device
    K, n = z.shape
    eps2 = eps * eps

    mu = mu.to(device).clamp_min(0.0)
    w = z.abs().square()
    q0 = (w * mu.unsqueeze(0)).sum(dim=-1)

    inside = q0 <= eps2
    lam = torch.zeros((K,), device=device, dtype=torch.float32)

    if inside.all():
        return lam

    idx = (~inside).nonzero(as_tuple=False).squeeze(-1)
    w_sub = w.index_select(0, idx)
    Kout = w_sub.shape[0]
    mu_row = mu.unsqueeze(0)

    def q_of(lam_vec):
        denom2 = (1.0 + 2.0 * lam_vec.unsqueeze(-1) * mu_row).square()
        return (mu_row * w_sub / denom2).sum(dim=-1)

    lo = torch.zeros((Kout,), device=device, dtype=torch.float32)
    hi = torch.ones((Kout,), device=device, dtype=torch.float32)

    q_hi = q_of(hi)
    _bracket_iters = 0
    for _ in range(40):
        need = q_hi > eps2
        if not need.any():
            break
        hi[need] *= 2.0
        q_hi = q_of(hi)
        _bracket_iters += 1
    else:
        raise RuntimeError("Failed to bracket lambda. Check M PSD and eps > 0.")

    _bisect_iters = 0
    for _bisect_iters in range(max_iter):
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


def _lambda_bisection_2d(
    w: torch.Tensor,
    mu_outer: torch.Tensor,
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
    _diag: bool = False,
) -> torch.Tensor:
    device = w.device
    K = w.shape[0]
    eps2 = eps * eps
    mu_outer = mu_outer.to(device).clamp_min(0.0)

    q0 = (w * mu_outer.unsqueeze(0)).sum(dim=(-2, -1))
    inside = q0 <= eps2
    lam = torch.zeros((K,), device=device, dtype=torch.float32)

    if _diag:
        print(f"      [BISECT2D DIAG] K={K}, eps={eps:.6f}, eps2={eps2:.4e}")
        print(
            f"      [BISECT2D DIAG] q0 range=[{q0.min().item():.4e}, {q0.max().item():.4e}]"
        )
        print(
            f"      [BISECT2D DIAG] #feasible={int(inside.sum().item())}/{K}, #infeasible={int((~inside).sum().item())}/{K}"
        )

    if inside.all():
        return lam

    idx = (~inside).nonzero(as_tuple=False).squeeze(-1)
    w_sub = w.index_select(0, idx)
    Kout = w_sub.shape[0]
    mu_row = mu_outer.unsqueeze(0)

    def q_of(lam_vec):
        denom2 = (1.0 + 2.0 * lam_vec.view(Kout, 1, 1) * mu_row).square()
        return (mu_row * w_sub / denom2).sum(dim=(-2, -1))

    lo = torch.zeros((Kout,), device=device, dtype=torch.float32)
    hi = torch.ones((Kout,), device=device, dtype=torch.float32)
    q_hi = q_of(hi)
    for _ in range(40):
        need = q_hi > eps2
        if not need.any():
            break
        hi[need] *= 2.0
        q_hi = q_of(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        q_mid = q_of(mid)
        right = q_mid <= eps2
        hi = torch.where(right, mid, hi)
        lo = torch.where(right, lo, mid)
        if (q_mid - eps2).abs().max().item() < tol:
            break

    lam.index_copy_(0, idx, hi)
    return lam


def project_factorized_2d(
    delta_flat: torch.Tensor,
    U: torch.Tensor,
    mu: torch.Tensor,
    H: int,
    W: int,
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
    _diag: bool = False,
) -> tuple:
    K = delta_flat.shape[0]
    device = delta_flat.device
    dtype = U.dtype

    delta_2d = delta_flat.to(dtype).reshape(K, H, W)

    z = torch.einsum("ip,kij,jq->kpq", U.conj(), delta_2d, U.conj())

    mu_outer = torch.outer(mu.to(device), mu.to(device))
    w = z.abs().square()

    if _diag:
        _q_pre = (mu_outer.unsqueeze(0) * w).sum(dim=(-2, -1))
        print(f"    [PROJ2D DIAG] eps={eps:.6f}, eps^2={eps**2:.4e}, K={K}")
        print(
            f"    [PROJ2D DIAG] q_pre: range=[{_q_pre.min().item():.4e}, {_q_pre.max().item():.4e}], #infeasible={int((_q_pre > eps**2).sum().item())}/{K}"
        )

    lam = _lambda_bisection_2d(
        w, mu_outer, eps, max_iter=max_iter, tol=tol, _diag=_diag
    )

    inv = 1.0 / (1.0 + 2.0 * lam.view(K, 1, 1) * mu_outer.unsqueeze(0))
    z_proj = z * inv

    if _diag:
        w_proj = z_proj.abs().square()
        _q_post = (mu_outer.unsqueeze(0) * w_proj).sum(dim=(-2, -1))
        print(
            f"    [PROJ2D DIAG] q_post: range=[{_q_post.min().item():.4e}, {_q_post.max().item():.4e}], eps^2={eps**2:.4e}, feasible={bool((_q_post <= eps**2 * 1.01).all().item())}"
        )

    proj_2d = torch.einsum("ip,kpq,jq->kij", U, z_proj, U)

    return proj_2d.real.reshape(K, H * W), lam


def project_CM_fast_eig(
    delta_flat: torch.Tensor,
    U: torch.Tensor,
    mu: torch.Tensor,
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = delta_flat.device
    delta_c = delta_flat.to(U.dtype)

    z = torch.matmul(delta_c, U)

    lam = _lambda_bisection_from_eigcoords(z, mu, eps, max_iter=max_iter, tol=tol)

    inv = 1.0 / (
        1.0 + 2.0 * lam.unsqueeze(-1) * mu.to(device).unsqueeze(0)
    )
    proj_c = torch.matmul(z * inv, U.conj().T)

    return proj_c.real, lam


def project_CM_batched_solve(
    delta_flat: torch.Tensor,
    Mherm: torch.Tensor,
    eps: float,
    max_iter: int = 25,
    tol: float = 1e-5,
    bracket_iters: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = delta_flat.device
    K, n = delta_flat.shape
    eps2 = eps * eps

    dtype = Mherm.dtype
    delta_c = delta_flat.to(dtype)

    M = 0.5 * (Mherm + Mherm.mH)

    q0 = torch.einsum("kn,nm,km->k", delta_c.conj(), M, delta_c).real
    inside = q0 <= eps2

    lam = torch.zeros((K,), device=device, dtype=torch.float32)
    proj = delta_c.clone()

    if inside.all():
        return delta_flat, lam

    idx = (~inside).nonzero(as_tuple=False).squeeze(-1)
    d = delta_c.index_select(0, idx)
    Kout = d.shape[0]

    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    Mbat = M.unsqueeze(0)

    def solve_and_q(lam_vec: torch.Tensor):
        lamv = lam_vec.to(dtype=dtype)
        A = I + (2.0 * lamv).view(Kout, 1, 1) * Mbat
        x = torch.linalg.solve(A, d.unsqueeze(-1)).squeeze(-1)
        q = torch.einsum("kn,nm,km->k", x.conj(), M, x).real
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
