import numpy as np

def apply_vignette(np_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Apply vignette effect.
    strength: 0.0 (no effect) to 1.0 (strong vignette)
    """
    H, W = np_rgb.shape[:2]
    Y, X = np.ogrid[:H, :W]
    cy, cx = H / 2, W / 2
    norm = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    mask = 1 - strength * (norm / max_dist)
    mask = np.clip(mask, 0, 1)
    vignette = np_rgb * mask[..., None]
    return vignette