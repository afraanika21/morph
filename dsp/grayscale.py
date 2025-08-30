import numpy as np

def blend_grayscale(np_rgb: np.ndarray, strength: float) -> np.ndarray:
    """
    strength: 0.0 -> original, 1.0 -> full grayscale
    grayscale = 0.299 R + 0.587 G + 0.114 B (luma approximation)
    """
    r, g, b = np_rgb[..., 0], np_rgb[..., 1], np_rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray3 = np.stack([gray, gray, gray], axis=-1)
    out = (1.0 - strength) * np_rgb + strength * gray3
    return out
    