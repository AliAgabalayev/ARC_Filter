import numpy as np
from scipy.special import erf

def ARC_filter(img: np.ndarray) -> np.ndarray:
    """
    Adaptive Rank-Cluster (ARC-) filter optimized for Python/NumPy.
    Input 'img' must be a 2D float64 NumPy array in the strict range [0..1].
    """
    eps_safe = 1e-16
    alpha = 1.0 / np.sqrt(2.0)
    beta = 1.0 / np.sqrt(2.0 * np.pi)
    rows, cols = img.shape
    
    # 1. Zero-copy symmetric padding and ultra-fast 3x3 sliding window vectorization
    padded = np.pad(img, pad_width=1, mode='symmetric')
    
    # Use NumPy stride tricks to simulate im2col without allocating new memory
    shape = (rows, cols, 3, 3)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    
    # Flatten spatial dimensions to match vector structure (9, num_windows)
    B = windows.transpose(2, 3, 0, 1).reshape(9, -1)
    center_vals = B[4, :].copy()
    
    # Fast in-place sorting along the window axis (axis=0)
    B.sort(axis=0)
    total_mu = B[4, :].copy()
    
    # Truncate the sorted set to 7 elements (rows 1..7 in 0-indexed Python)
    B_trunc = B[1:8, :].copy()
    del B
    
    # 2. Otsu thresholding across 6 possible cuts (0..5 in 0-indexed Python)
    cumsum_F = np.cumsum(B_trunc, axis=0)
    sum_total_7 = cumsum_F[6, :].copy()
    
    buf_A = cumsum_F[0:6, :].copy()
    del cumsum_F
    
    # Matrix-free Otsu broadcasting optimized for CPU cache
    cuts_idx = np.arange(1, 7, dtype=np.float64)[:, np.newaxis]
    MV = np.array([1/3, 1/5, 1/6, 1/6, 1/5, 1/3], dtype=np.float64)[:, np.newaxis]
    
    V = MV * (buf_A * 7.0 - cuts_idx * sum_total_7) ** 2
    N1 = np.argmax(V, axis=0) + 1.0  # Convert 0-indexed to 1-based size count
    N2 = 7.0 - N1
    del V
    
    # Fast 1D advanced indexing using NumPy arrays
    num_windows = rows * cols
    lin_opt_rows = (N1 - 1).astype(np.intp) 
    lin_opt_cols = np.arange(num_windows, dtype=np.intp)
    
    selected_buf_A = buf_A[lin_opt_rows, lin_opt_cols]
    final_mu1 = selected_buf_A / N1
    final_mu2 = (sum_total_7 - selected_buf_A) / N2
    del buf_A
    
    # 3. Standard deviation calculation via single sum of squares
    cumsum_Sq = np.cumsum(B_trunc ** 2, axis=0)
    del B_trunc
    
    buf_SqA = cumsum_Sq[0:6, :].copy()
    sum_sq1_opt = buf_SqA[lin_opt_rows, lin_opt_cols]
    del buf_SqA
    
    sum_sq2_opt = cumsum_Sq[6, :] - sum_sq1_opt
    del cumsum_Sq, lin_opt_rows, lin_opt_cols
    
    # Unbiased standard deviation formula
    sigma1 = np.maximum(eps_safe, np.sqrt(np.abs(N1 * sum_sq1_opt - (N1 * final_mu1) ** 2)) / N1)
    sigma2 = np.maximum(eps_safe, np.sqrt(np.abs(N2 * sum_sq2_opt - (N2 * final_mu2) ** 2)) / N2)
    del sum_sq1_opt, sum_sq2_opt, N1, N2
    
    # 4. Separation threshold (Sth) computation and soft fuzzy transition
    inv_sigma_dist = 1.0 / (sigma1 + sigma2)
    mu_dist = np.maximum(1.0, (final_mu2 - final_mu1) * inv_sigma_dist)
    
    Sth = (final_mu1 * sigma2 + final_mu2 * sigma1) * inv_sigma_dist
    del final_mu1, final_mu2, inv_sigma_dist
    
    msk = (center_vals < Sth)
    sigma_tot = np.where(msk, sigma1, sigma2)
    del sigma1, sigma2, msk
    
    res = Sth + sigma_tot * mu_dist * erf(beta * (center_vals - Sth) / sigma_tot)
    del sigma_tot, center_vals, Sth
    
    # 5. Soft cluster mixing and reshaping back to 2D matrix layout
    out_flat = total_mu + erf(np.abs(alpha * (mu_dist - 1.0))) * (res - total_mu)
    del mu_dist, total_mu, res
    
    return out_flat.reshape(rows, cols)