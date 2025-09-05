# dsp/dct.py
import numpy as np
from typing import Tuple

# -----------------------------
# 0) Color space helpers (RGB <-> YCbCr)
# -----------------------------
def _rgb_to_ycbcr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """img: HxWx3 float32 in [0,255] -> returns (Y, Cb, Cr) each HxW float32."""
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    # ITU-R BT.601
    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cb = -0.168736* R - 0.331264* G + 0.5     * B + 128.0
    Cr =  0.5     * R - 0.418688* G - 0.081312* B + 128.0
    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)

def _ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """Y, Cb, Cr: HxW float32 -> returns HxWx3 float32 in [0,255] (clipped later)."""
    R = Y + 1.402   * (Cr - 128.0)
    G = Y - 0.344136* (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772   * (Cb - 128.0)
    out = np.stack([R, G, B], axis=-1).astype(np.float32)
    return out

# -----------------------------
# 1) DCT-II / IDCT-III basis (N x N)
# -----------------------------
def _dct_matrix(N: int) -> np.ndarray:
    """Return NxN DCT-II transform matrix with orthonormal scaling."""
    C = np.zeros((N, N), dtype=np.float32)
    factor = np.pi / (2.0 * N)
    scale0 = np.sqrt(1.0 / N)
    scale  = np.sqrt(2.0 / N)

    for k in range(N):
        alpha = scale0 if k == 0 else scale
        for n in range(N):
            C[k, n] = alpha * np.cos((2*n + 1) * k * factor)
    return C  # DCT-II basis

def _dct2(block: np.ndarray, C: np.ndarray) -> np.ndarray:
    """2D DCT via separable transforms: C * block * C^T"""
    return (C @ block @ C.T).astype(np.float32)

def _idct2(coeff: np.ndarray, C: np.ndarray) -> np.ndarray:
    """2D IDCT for orthonormal DCT: C^T * coeff * C"""
    return (C.T @ coeff @ C).astype(np.float32)

# -----------------------------
# 2) Block tiling helpers
# -----------------------------
def _pad_to_block(img: np.ndarray, N: int) -> Tuple[np.ndarray, int, int]:
    H, W = img.shape
    padH = (N - (H % N)) % N
    padW = (N - (W % N)) % N
    if padH or padW:
        img = np.pad(img, ((0, padH), (0, padW)), mode="edge")
    return img, H, W  # keep original size for unpad

def _unpad(img: np.ndarray, H: int, W: int) -> np.ndarray:
    return img[:H, :W]

# -----------------------------
# 3) Zigzag order (keeps low freqs first)
# -----------------------------
def _zigzag_indices(N: int) -> np.ndarray:
    idx = []
    for s in range(2*N - 1):
        if s % 2 == 0:
            i = min(s, N-1)
            j = s - i
            while i >= 0 and j < N:
                idx.append((i, j))
                i -= 1; j += 1
        else:
            j = min(s, N-1)
            i = s - j
            while j >= 0 and i < N:
                idx.append((i, j))
                i += 1; j -= 1
    return np.array(idx, dtype=np.int32)

# -----------------------------
# 4) JPEG quantization tables + scaling
# -----------------------------
_QY_BASE = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

_QC_BASE = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.float32)

def _scale_qtable(Q: np.ndarray, quality: int) -> np.ndarray:
    """JPEG quality scaling: 1(worst)..100(best)."""
    q = int(round(quality))
    q = max(1, min(100, q))
    if q < 50:
        S = 5000 / q
    else:
        S = 200 - 2*q
    Qs = np.floor((Q * S + 50) / 100).astype(np.float32)
    Qs = np.clip(Qs, 1, 255)
    return Qs

# -----------------------------
# 5) Public APIs
# -----------------------------
def apply_dct_jpeg(np_rgb: np.ndarray, quality: int = 75, block: int = 8) -> np.ndarray:
    """
    JPEG-like compression:
      - Convert to YCbCr
      - Level shift by 128
      - 2D DCT on NxN blocks
      - Quantize / Dequantize with scaled tables (quality 1..100)
      - Inverse DCT
      - Add 128, convert back to RGB
    NOTE: JPEG quant tables are 8x8. For block != 8 we fall back to using 8x8 internally per tile.
    """
    img = np_rgb.astype(np.float32)
    Y, Cb, Cr = _rgb_to_ycbcr(img)

    # Use 8x8 for quantization (standard). If block != 8, we still process in 8x8 tiles.
    N = 8
    C = _dct_matrix(N)
    QY = _scale_qtable(_QY_BASE, quality)
    QC = _scale_qtable(_QC_BASE, quality)

    def _proc_channel(chan: np.ndarray, Q: np.ndarray) -> np.ndarray:
        ch = chan - 128.0
        ch, H, W = _pad_to_block(ch, N)
        out = np.empty_like(ch)
        for y in range(0, ch.shape[0], N):
            for x in range(0, ch.shape[1], N):
                block = ch[y:y+N, x:x+N]
                dctb  = _dct2(block, C)
                # Quantize -> Dequantize
                q     = np.round(dctb / Q)
                deq   = q * Q
                out[y:y+N, x:x+N] = _idct2(deq, C)
        out = _unpad(out, H, W) + 128.0
        return out

    Yq  = _proc_channel(Y,  QY)
    Cbq = _proc_channel(Cb, QC)
    Crq = _proc_channel(Cr, QC)
    out = _ycbcr_to_rgb(Yq, Cbq, Crq)
    return np.clip(out, 0.0, 255.0).astype(np.float32)

def apply_dct_keep(np_rgb: np.ndarray, keep: float = 0.125, block: int = 8,
                   mode: str = "zigzag") -> np.ndarray:
    """
    Keep a fraction of DCT coefficients per block and zero-out the rest.
    mode: 'zigzag' (keep low frequencies) or 'topk' (keep largest magnitudes).
    keep: 0<keep<=1  (e.g., 0.125 keeps 8 of 64 coeffs in 8x8)
    Operates per-channel in RGB for simplicity.
    """
    img = np_rgb.astype(np.float32)
    H, W, _ = img.shape
    N = int(block)
    N = N if N > 0 else 8
    C = _dct_matrix(N)
    out = np.empty_like(img)

    if mode not in ("zigzag", "topk"):
        mode = "zigzag"

    if mode == "zigzag":
        order = _zigzag_indices(N)
        K = max(1, min(N*N, int(round(keep * (N*N)))))
    else:
        K = max(1, min(N*N, int(round(keep * (N*N)))))

    def _proc_block(b: np.ndarray) -> np.ndarray:
        d = _dct2(b, C)
        if mode == "zigzag":
            mask = np.zeros_like(d, dtype=bool)
            # Keep first K positions in zigzag order
            for i in range(K):
                yy, xx = order[i]
                mask[yy, xx] = True
            d = np.where(mask, d, 0.0)
        else:
            # Keep top-K magnitudes
            flat = d.reshape(-1)
            idx = np.argpartition(np.abs(flat), -K)[-K:]
            mask = np.zeros_like(flat, dtype=bool)
            mask[idx] = True
            d = (flat * mask).reshape(d.shape)
        return _idct2(d, C)

    # per channel
    for c in range(3):
        ch, HH, WW = _pad_to_block(img[..., c], N)
        tmp = np.empty_like(ch)
        for y in range(0, ch.shape[0], N):
            for x in range(0, ch.shape[1], N):
                tmp[y:y+N, x:x+N] = _proc_block(ch[y:y+N, x:x+N])
        out[..., c] = _unpad(tmp, HH, WW)

    return np.clip(out, 0.0, 255.0).astype(np.float32)

# -----------------------------
# 6) PSNR metric
# -----------------------------
def psnr(orig: np.ndarray, comp: np.ndarray) -> float:
    """Peak SNR in dB for 8-bit images. Returns +inf if identical."""
    x = orig.astype(np.float32)
    y = comp.astype(np.float32)
    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)
