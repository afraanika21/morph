# dsp/noise.py
import numpy as np

def _rng(seed):
    return np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

def to_float(img):
    # expects HxWxC uint8 or float
    x = img.astype(np.float64, copy=False)
    return np.clip(x, 0, 255)

def from_float(x):
    return np.clip(x, 0, 255).astype(np.uint8)

# -------------------- Noise models --------------------

def add_awgn(img_rgb, sigma=20.0, seed=None):
    """
    Additive white Gaussian noise per channel.
    sigma in [0..50] on 0..255 scale.
    """
    x = to_float(img_rgb)
    r = _rng(seed)
    noisy = x + r.normal(loc=0.0, scale=float(sigma), size=x.shape)
    return from_float(noisy)

def add_salt_pepper(img_rgb, p=0.02, seed=None):
    """
    Flip random pixels to 0 or 255 with prob p (per pixel, per channel).
    Best seen on darker images; median is effective against it.
    """
    x = to_float(img_rgb)
    r = _rng(seed)
    mask = r.random(size=x.shape[:2])  # one mask per pixel, broadcast to channels
    sp = x.copy()
    salt = mask < (p / 2.0)
    pepper = (mask >= (p / 2.0)) & (mask < p)
    sp[salt, :] = 255
    sp[pepper, :] = 0
    return from_float(sp)

def add_speckle(img_rgb, sigma=0.15, seed=None):
    """
    Multiplicative noise: y = x * (1 + n), n~N(0, sigma^2)
    """
    x = to_float(img_rgb)
    r = _rng(seed)
    n = r.normal(loc=0.0, scale=float(sigma), size=x.shape)
    y = x * (1.0 + n)
    return from_float(y)

def add_poisson(img_rgb, scale=12.0, seed=None):
    """
    Poisson noise: treat intensities as photon counts (scaled).
    Larger 'scale' => higher SNR (more counts).
    """
    x = to_float(img_rgb)
    lam = np.clip(x * (float(scale) / 255.0), 0, None)
    r = _rng(seed)
    k = r.poisson(lam=lam)
    y = k * (255.0 / max(scale, 1e-6))
    return from_float(y)
