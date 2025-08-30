# dsp/freq.py
import numpy as np

def _pad_reflect2d(a: np.ndarray, py: int, px: int) -> np.ndarray:
    if py == 0 and px == 0: return a
    return np.pad(a, ((py, py), (px, px)), mode="reflect")

def _unpad_center(a: np.ndarray, H: int, W: int) -> np.ndarray:
    y0 = (a.shape[0] - H)//2
    x0 = (a.shape[1] - W)//2
    return a[y0:y0+H, x0:x0+W]

def _gaussian_lpf_mask(h: int, w: int, cutoff: float) -> np.ndarray:
    """
    cutoff: 0..0.5 (as a fraction of Nyquist). 0.25 ≈ gentle blur, 0.05 ≈ heavy blur.
    """
    # radial frequency grid (normalized to Nyquist radius = 0.5)
    cy, cx = h/2.0, w/2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(X*X + Y*Y)
    r_norm = r / (0.5 * np.sqrt(h*h + w*w))  # 0 at center, 1 at “corners” (Nyquist-ish)
    # Gaussian in frequency: exp(-(r^2)/(2*sigma_f^2)), map cutoff to sigma_f
    sigma_f = max(1e-6, float(cutoff))
    mask = np.exp(-(r_norm**2) / (2.0 * sigma_f * sigma_f))
    return mask.astype(np.float32)

def _apply_fft_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)
    Ff = F * mask
    Ff = np.fft.ifftshift(Ff)
    out = np.fft.ifft2(Ff).real.astype(np.float32)
    return out

def _rgb_to_float32(np_rgb: np.ndarray) -> np.ndarray:
    return np_rgb.astype(np.float32)

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 255.0)

def apply_lowpass_fft(np_rgb: np.ndarray, cutoff: float = 0.15, pad: bool = True) -> np.ndarray:
    """
    Frequency-domain Gaussian LPF on each channel.
    cutoff in (0..0.5]: smaller = more blur.
    """
    img = _rgb_to_float32(np_rgb)
    H, W = img.shape[:2]
    py, px = (H//2 if pad else 0), (W//2 if pad else 0)

    out = np.empty_like(img)
    # build mask once for padded size
    hP, wP = H + 2*py, W + 2*px
    mask = _gaussian_lpf_mask(hP, wP, cutoff=cutoff)

    for c in range(3):
        ch = _pad_reflect2d(img[..., c], py, px)
        ch_lpf = _apply_fft_mask(ch, mask)
        ch_lpf = _unpad_center(ch_lpf, H, W)
        out[..., c] = ch_lpf
    return _clip01(out)

def apply_highpass_fft(np_rgb: np.ndarray,
                       cutoff: float = 0.15,
                       highboost: float = 0.0,
                       pad: bool = True) -> np.ndarray:
    """
    HPF via (I - LPF(I)). If highboost>0, returns I + highboost*HPF.
    cutoff: same as LPF (smaller → stronger HP).
    """
    img = _rgb_to_float32(np_rgb)
    lpf = apply_lowpass_fft(img, cutoff=cutoff, pad=pad)
    hpf = img - lpf
    if highboost and float(highboost) != 0.0:
        out = img + float(highboost) * hpf
    else:
        # visualize pure high-pass (shift to mid-gray for visibility)
        out = (hpf + 128.0)
    return _clip01(out)
