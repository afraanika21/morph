# dsp/tonemap.py
import numpy as np

def apply_brightness_contrast(np_rgb: np.ndarray, brightness_pct: float = 0.0, contrast_pct: float = 0.0) -> np.ndarray:
    """
    brightness_pct: [-100..100] -> beta shift (additive)
    contrast_pct:   [-100..100] -> alpha gain (multiplicative around mid-gray 128)

    out = alpha * (img - 128) + 128 + beta
    where alpha = 1 + contrast_pct/100, beta = brightness_pct
    """
    alpha = 1.0 + (contrast_pct / 100.0)
    beta = brightness_pct

    out = alpha * (np_rgb - 128.0) + 128.0 + beta
    return out
