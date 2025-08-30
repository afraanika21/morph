# dsp/pipeline.py
from typing import Dict, Any
import numpy as np
from dsp.grayscale import blend_grayscale
from dsp.tonemap import apply_brightness_contrast
from dsp.conv import apply_gaussian_blur
from dsp.vignette import apply_vignette

def apply_pipeline(np_rgb: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Optional params:
      - grayscale_strength: 0..100
      - blur_sigma: 0..5
      - brightness: -100..100
      - contrast:   -100..100
      - vignette_strength: 0..1   # <--- add this

    Order: grayscale -> blur -> brightness/contrast -> vignette
    """
    out = np_rgb

    gs = params.get("grayscale_strength", None)
    if gs is not None:
        strength = max(0.0, min(100.0, float(gs))) / 100.0
        out = blend_grayscale(out, strength)

    bs = params.get("blur_sigma", None)
    if bs is not None:
        sigma = max(0.0, min(5.0, float(bs)))
        out = apply_gaussian_blur(out, sigma)

    br = params.get("brightness", None)
    ct = params.get("contrast", None)
    if br is not None or ct is not None:
        br = 0.0 if br is None else max(-100.0, min(100.0, float(br)))
        ct = 0.0 if ct is None else max(-100.0, min(100.0, float(ct)))
        out = apply_brightness_contrast(out, brightness_pct=br, contrast_pct=ct)

    # --- Vignette ---
    vs = params.get("vignette_strength", None)
    print("Vignette strength received:", vs)  # <-- Add this line
    if vs is not None:
        strength = max(0.0, min(1.0, float(vs)))
        out = apply_vignette(out, strength)

    return out
