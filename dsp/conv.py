# dsp/conv.py
import numpy as np

def _gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0:
        # no blur → identity 1x1 kernel
        return np.array([[1.0]], dtype=np.float32)
    # Kernel half-width: 3*sigma (classic rule-of-thumb)
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g1d = np.exp(-(x**2) / (2.0 * sigma * sigma))
    g1d /= g1d.sum()
    # Separable: 1D conv horizontally then vertically
    return g1d  # return 1D gaussian for separable conv

def _pad_reflect(arr: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return arr
    return np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")

def _conv1d_sep_channel(chan: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Separable 2-pass conv (reflect padding). chan: HxW float32."""
    r = (len(g) - 1) // 2
    # horizontal
    ph = _pad_reflect(chan, r)
    H, W = chan.shape
    tmp = np.empty_like(chan)
    for y in range(H):
        row = ph[y + r, :]  # pick padded row
        acc = np.zeros(W, dtype=np.float32)
        # slide window via convolution (vectorized with stride add)
        for k in range(-r, r + 1):
            acc += g[k + r] * ph[y + r, (r + k):(r + k + W)]
        tmp[y, :] = acc
    # vertical
    pv = _pad_reflect(tmp.T, r)  # transpose to reuse same loop
    out_t = np.empty_like(tmp.T)
    for x in range(W):
        col = pv[x + r, :]
        acc = np.zeros(H, dtype=np.float32)
        for k in range(-r, r + 1):
            acc += g[k + r] * pv[x + r, (r + k):(r + k + H)]
        out_t[x, :] = acc
    return out_t.T

def apply_gaussian_blur(np_rgb: np.ndarray, sigma: float) -> np.ndarray:
    """
    np_rgb: HxWx3 float32 in [0,255]
    sigma: 0..5 (0 means no blur)
    """
    g = _gaussian_kernel(float(sigma))
    if g.shape[0] == 1:  # sigma==0
        return np_rgb
    r = (len(g) - 1) // 2
    out = np.empty_like(np_rgb)
    # process each channel
    for c in range(3):
        out[..., c] = _conv1d_sep_channel(np_rgb[..., c], g)
    return out
