# dsp/denoise.py
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import convolve2d

EPS = 1e-8

# -------------------- Colorspace helpers --------------------

def rgb_to_ycbcr(rgb):
    # ITU-R BT.601 (SDTV)
    x = rgb.astype(np.float64)
    r, g, b = x[...,0], x[...,1], x[...,2]
    y  =  0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 128.0
    cr =  0.5*r - 0.418688*g - 0.081312*b + 128.0
    return np.stack([y, cb, cr], axis=-1)

def ycbcr_to_rgb(ycc):
    y, cb, cr = ycc[...,0], ycc[...,1], ycc[...,2]
    r = y + 1.402*(cr-128.0)
    g = y - 0.344136*(cb-128.0) - 0.714136*(cr-128.0)
    b = y + 1.772*(cb-128.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)

# -------------------- Denoisers --------------------

def moving_average(img_rgb, k=3):
    """
    Box/mean filter via uniform_filter. k must be odd (3,5,7).
    """
    k = max(1, int(k))
    if k % 2 == 0: k += 1
    out = np.zeros_like(img_rgb, dtype=np.float64)
    for c in range(3):
        out[..., c] = uniform_filter(img_rgb[..., c].astype(np.float64), size=k, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)

def median_denoise(img_rgb, k=3):
    """
    Channel-wise median filtering. Great for salt & pepper.
    """
    k = max(1, int(k))
    if k % 2 == 0: k += 1
    out = np.zeros_like(img_rgb, dtype=np.float64)
    for c in range(3):
        out[..., c] = median_filter(img_rgb[..., c].astype(np.float64), size=k, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_float(img):
    # expects HxWxC or HxW, converts to float64 [0..255] range
    arr = np.asarray(img, dtype=np.float64)
    return arr

def _ensure_3c(img):
    if img.ndim == 2:
        return img[..., None]
    return img

def _gaussian_psf(size: int, sigma: float):
    size = int(max(3, size | 1))      # odd
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum() + EPS
    return psf

def _otf_from_psf(psf, shape_hw):
    """Pad a small PSF to image size and FFT -> OTF. No shifts needed."""
    H, W = shape_hw
    Ph, Pw = psf.shape
    big = np.zeros((H, W), dtype=np.float64)
    # place psf at top-left, but circularly shift to center-of-kernel at (0,0)
    # this 'wrap' aligns the effective impulse at origin for FFT.
    i0 = (H - Ph) // 2
    j0 = (W - Pw) // 2
    big[i0:i0+Ph, j0:j0+Pw] = psf
    # circularly roll so that kernel center is at (0,0)
    big = np.roll(big, -Ph//2, axis=0)
    big = np.roll(big, -Pw//2, axis=1)
    return fft2(big)

def wiener_fft_rgb(img_rgb, nsr=0.01, psf_size=None, psf_sigma=None):
    """
    Frequency-domain Wiener:
      - If psf_sigma is None: denoising (identity system).
      - Else: deblurring with Gaussian PSF(psf_size, psf_sigma).
    nsr = noise-to-signal ratio (scalar). Typical 0.005..0.05
    """
    X = _to_float(img_rgb)
    X = _ensure_3c(X)
    H, W, C = X.shape

    if psf_sigma is None or (psf_size is None):
        # identity H => Wiener denoising: F_hat = G * (1 / (1 + NSR))
        # (equivalent to X * (|H|^2 / (|H|^2+K)) with H=1)
        G = fft2(X, axes=(0,1))
        K = float(max(nsr, 0.0))
        Fhat = G * (1.0 / (1.0 + K))
        out = np.real(ifft2(Fhat, axes=(0,1)))
        out = np.clip(out, 0, 255)
        return out.squeeze() if C == 1 else out

    # Deblurring Wiener with Gaussian PSF
    psf = _gaussian_psf(int(psf_size), float(psf_sigma))
    Hf = _otf_from_psf(psf, (H, W))                       # (H,W) complex
    Hpow = (Hf.real**2 + Hf.imag**2)                      # |H|^2
    denom = Hpow + float(nsr)                             # |H|^2 + K
    denom[denom < EPS] = EPS

    out = np.zeros_like(X, dtype=np.float64)
    for ch in range(C):
        G = fft2(X[..., ch])
        Fhat = (np.conj(Hf) * G) / denom                  # H* / (|H|^2 + K) * G
        out[..., ch] = np.real(ifft2(Fhat))

    out = np.clip(out, 0, 255)
    return out.squeeze() if C == 1 else out


# -------------------- Metrics --------------------

def psnr(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((ref - test)**2)
    if mse <= 1e-12:
        return 99.0
    max_i = 255.0
    return 10.0 * np.log10((max_i*max_i) / mse)

def _ssim_channel(x, y, C1, C2):
    mu_x = uniform_filter(x, size=7)
    mu_y = uniform_filter(y, size=7)
    sigma_x2 = uniform_filter(x*x, size=7) - mu_x*mu_x
    sigma_y2 = uniform_filter(y*y, size=7) - mu_y*mu_y
    sigma_xy = uniform_filter(x*y, size=7) - mu_x*mu_y

    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
    s = num / (den + 1e-12)
    return np.mean(s)

def ssim(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    K1, K2 = 0.01, 0.03
    L = 255.0
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    s = 0.0
    for c in range(3):
        s += _ssim_channel(ref[...,c], test[...,c], C1, C2)
    return s / 3.0
