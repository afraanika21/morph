# dsp/edges.py
import numpy as np
from typing import Tuple
from dsp.conv import apply_gaussian_blur
from numpy.lib.stride_tricks import sliding_window_view

def _to_gray(np_rgb: np.ndarray) -> np.ndarray:
    r, g, b = np_rgb[...,0], np_rgb[...,1], np_rgb[...,2]
    return 0.299*r + 0.587*g + 0.114*b

def _pad_reflect(a: np.ndarray, py: int, px: int) -> np.ndarray:
    return np.pad(a, ((py, py), (px, px)), mode="reflect")

def _conv2d_gray(gray: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Vectorized 2D conv for tiny kernels (e.g., 3x3). Reflect padding."""
    kh, kw = k.shape
    ry, rx = kh // 2, kw // 2
    p = np.pad(gray, ((ry, ry), (rx, rx)), mode="reflect")
    # windows: (H, W, kh, kw)
    windows = sliding_window_view(p, (kh, kw))
    # einsum multiplies each (kh,kw) window by k and sums -> (H, W)
    out = np.einsum('ijkl,kl->ij', windows, k, optimize=True)
    return out.astype(np.float32)



def _normalize_0_255(x: np.ndarray) -> np.ndarray:
    m = x.max()
    if m <= 1e-8: 
        return np.zeros_like(x, dtype=np.float32)
    return (x / m) * 255.0

def _overlay(np_rgb: np.ndarray, edge_mask: np.ndarray,
             color: tuple[float, float, float], alpha: float) -> np.ndarray:
    """Vectorized overlay of colored edges."""
    out = np_rgb.astype(np.float32)
    mask3 = edge_mask[..., None]  # HxWx1
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.where(mask3,
                    (1.0 - alpha) * out + alpha * color_arr,
                    out)

def apply_edges(np_rgb: np.ndarray,
                kind: str = "sobel",
                threshold: float = 100.0,
                overlay: bool = False,
                overlay_alpha: float = 0.6,
                overlay_color: Tuple[int,int,int] = (255, 64, 64),
                smooth_sigma: float = 1.0,
                return_map: bool = False) -> np.ndarray:
    """
    kind: 'sobel' | 'prewitt' | 'laplacian'
    threshold: 0..255 applied after normalization
    overlay: draw colored edges over original
    return_map: if True, return edge map as RGB (no overlay), else use overlay flag
    """
    img = np_rgb.astype(np.float32)
    if smooth_sigma and smooth_sigma > 0:
        img = apply_gaussian_blur(img, float(smooth_sigma))

    gray = _to_gray(img)

    if kind.lower() == "sobel":
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        ky = kx.T
        gx = _conv2d_gray(gray, kx)
        gy = _conv2d_gray(gray, ky)
        mag = np.sqrt(gx*gx + gy*gy)

    elif kind.lower() == "prewitt":
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
        ky = kx.T
        gx = _conv2d_gray(gray, kx)
        gy = _conv2d_gray(gray, ky)
        mag = np.sqrt(gx*gx + gy*gy)

    elif kind.lower() == "laplacian":
        # 8-neighbor Laplacian (stronger)
        k = np.array([[ -1,-1,-1],
                      [ -1, 8,-1],
                      [ -1,-1,-1]], dtype=np.float32)
        resp = np.abs(_conv2d_gray(gray, k))
        mag = resp
    else:
        raise ValueError("Unknown edge kind")

    m255 = _normalize_0_255(mag)
    edges_bin = (m255 >= float(threshold))

    if return_map and not overlay:
        # return grayscale edge map as RGB
        rgb = np.stack([m255, m255, m255], axis=-1)
        return rgb

    if overlay:
        alpha = max(0.0, min(1.0, float(overlay_alpha)))
        color = tuple(float(c) for c in overlay_color)
        return _overlay(np_rgb.astype(np.float32), edges_bin, color, alpha)

    # default: binary white edges on black (as RGB)
    edge_rgb = np.zeros_like(np_rgb, dtype=np.float32)
    for c in range(3):
        edge_rgb[..., c] = np.where(edges_bin, 255.0, 0.0)
    return edge_rgb
