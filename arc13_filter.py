"""Thirteen-position ARC variants; the original ARC9 remains in arc_filter.py."""
import numpy as np
from scipy.special import erf

OFFSETS13 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) + (
    (-2, 0), (2, 0), (0, -2), (0, 2))


def ARC13_filter(img: np.ndarray, trim: int = 2) -> np.ndarray:
    """Filter a 2D float64 [0,1] image, trimming 2 or 3 ranks at each end."""
    if trim not in (2, 3):
        raise ValueError("trim must be 2 or 3")
    return _filter(img, OFFSETS13, trim)


def _filter(img, offsets, trim):
    rows, cols = img.shape
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    padded = np.pad(img, radius, mode="symmetric")
    center = img.reshape(-1).copy()
    values = np.stack([padded[radius + dy:radius + dy + rows,
                              radius + dx:radius + dx + cols].reshape(-1)
                       for dy, dx in offsets])
    values.sort(axis=0)
    median = values[len(offsets) // 2].copy()
    retained = values[trim:len(offsets) - trim].copy()
    del values
    n = len(retained)
    sums = np.cumsum(retained, axis=0)
    k = np.arange(1, n, dtype=np.float64)[:, None]
    # Twice the usual Otsu objective, matching the ARC9 scaling exactly.
    objective = (2.0 / (k * (n - k))) * (n * sums[:-1] - k * sums[-1]) ** 2
    cut = np.argmax(objective, axis=0)
    n1 = cut + 1.0
    n2 = n - n1
    ix = np.arange(rows * cols)
    selected = sums[cut, ix]
    mu1 = selected / n1
    mu2 = (sums[-1] - selected) / n2
    squares = np.cumsum(retained ** 2, axis=0)
    sq1 = squares[cut, ix]
    sq2 = squares[-1] - sq1
    sigma1 = np.maximum(1e-16, np.sqrt(np.abs(n1 * sq1 - (n1 * mu1) ** 2)) / n1)
    sigma2 = np.maximum(1e-16, np.sqrt(np.abs(n2 * sq2 - (n2 * mu2) ** 2)) / n2)
    inverse = 1.0 / (sigma1 + sigma2)
    distance = np.maximum(1.0, (mu2 - mu1) * inverse)
    threshold = (mu1 * sigma2 + mu2 * sigma1) * inverse
    sigma = np.where(center < threshold, sigma1, sigma2)
    transformed = threshold + sigma * distance * erf(
        (center - threshold) / (np.sqrt(2.0 * np.pi) * sigma))
    result = median + erf(np.abs((distance - 1.0) / np.sqrt(2.0))) * (transformed - median)
    return result.reshape(rows, cols)
