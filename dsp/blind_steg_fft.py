
# dsp/blind_steg_fft.py
# Standard Blind Steganography Simulation Pipeline (DSP)
# - Bitstream encoding (sync, length, payload)
# - Key-based position selection (mid-band annulus)
# - QIM parity embedding/extraction
# - Repetition coding/majority vote
# - FFT domain (blue channel)

import hashlib
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# -------- Bit/stream helpers --------

HEADER_BITS = 32
# Barker-13 sync: [1,1,1,1,1,0,0,1,1,0,1,0,1]
SYNC = [1,1,1,1,1,0,0,1,1,0,1,0,1]

def _int_to_bits(x: int, width: int):
    s = format(int(x), f"0{int(width)}b")
    return [int(ch) for ch in s]

def _bits_to_int(bits):
    if not bits:
        return 0
    return int("".join("1" if b else "0" for b in bits), 2)

def text_to_bits_utf8(s: str):
    data = s.encode("utf-8")
    return [int(bit) for byte in data for bit in f"{byte:08b}"]

def bits_to_text_utf8(bits):
    if not bits:
        return ""
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    out = bytearray()
    for i in range(0, n, 8):
        byte = int("".join(str(b) for b in bits[i:i+8]), 2)
        out.append(byte)
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return "[decode error]"

def repeat_bits(bits, n):
    n = max(1, int(n))
    # FIXED: use 'bit' (not undefined 'b')
    return [bit for bit in bits for _ in range(n)]

def majority_vote(bits, n):
    n = max(1, int(n))
    if n == 1:
        return bits[:]
    out = []
    for i in range(0, len(bits), n):
        grp = bits[i:i+n]
        out.append(1 if sum(grp) >= (len(grp)/2.0) else 0)
    return out

def build_stream(text: str):
    """SYNC + 32-bit length (bytes) + UTF-8 payload bits."""
    payload = text_to_bits_utf8(text)
    L = len(payload) // 8
    return SYNC + _int_to_bits(L, HEADER_BITS) + payload

def find_sync(bits):
    m = len(SYNC)
    for i in range(0, len(bits) - m + 1):
        if bits[i:i+m] == SYNC:
            return i
    return -1

# -------- Keyed positions --------

def _rng_from_key(key: str):
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    return np.random.default_rng(seed)

def _annulus_mask(M, N, r_low, r_high):
    cx, cy = M//2, N//2
    Y, X = np.ogrid[:M, :N]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    inner, outer = maxd * float(r_low), maxd * float(r_high)
    return (dist >= inner) & (dist <= outer)

def select_positions_by_key(F, r_low, r_high, key, min_mag_percentile=30):
    mask = _annulus_mask(*F.shape, r_low, r_high)
    R, C = np.where(mask)
    if R.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    mags = np.abs(F[R, C])
    thr = np.percentile(mags, float(min_mag_percentile)) if mags.size else 0.0
    keep = mags >= max(thr, 1e-12)
    R, C = R[keep], C[keep]
    if R.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    idx = np.column_stack([R, C])
    rng = _rng_from_key(key)  # deterministic for a given key
    rng.shuffle(idx)
    return idx

# -------- QIM on log-magnitude parity --------

def _logmag(z):
    return np.log1p(np.abs(z))

def _set_parity_logmag(z, bit, step):
    """Quantize log|z| with step, set bin parity to 'bit' with minimal change."""
    mag = np.abs(z); phase = np.angle(z)
    lm = np.log1p(mag)
    q = np.floor(lm / float(step))
    want = int(bit) & 1
    if (int(q) & 1) != want:
        lo = (q - 1) * step
        hi = (q + 1) * step
        lm_new = lo if abs(lm - lo) <= abs(lm - hi) else hi
    else:
        lm_new = q * step
    mag_new = np.expm1(max(float(lm_new), 0.0))
    return mag_new * np.exp(1j * phase)

def _get_parity_logmag(z, step):
    q = np.floor(_logmag(z) / float(step))
    return int(q) & 1

# -------- Blind embed/extract --------

def _map_embed_ratio_to_band(embed_ratio: float):
    """Map a slider (0.1..0.9) to [r_low, r_high] mid-band annulus."""
    r_high = 0.88
    thickness = max(0.05, min(0.7, float(embed_ratio)))
    r_low = max(0.05, r_high - thickness)
    return r_low, r_high

def embed_blind_fft_rgb(cover_rgb, text, key,
                        repeat_n=3, embed_ratio=0.30,
                        step=0.05, min_mag_percentile=30):
    """
    Blind embedding: only stego + key are needed to extract.
    Uses log-magnitude QIM parity with key-driven positions.
    """
    # --- DSP Pipeline: Embed ---
    # 1. FFT (blue channel)
    blue = cover_rgb[:, :, 2].astype(np.float64, copy=True)
    M, N = blue.shape
    F = fftshift(fft2(blue)); Fst = F.copy()

    # 2. Bitstream: SYNC + length + payload, then repetition coding
    stream = build_stream(text)
    stream_rep = repeat_bits(stream, repeat_n)

    # 3. Key-based position selection (mid-band annulus)
    r_low, r_high = _map_embed_ratio_to_band(embed_ratio)
    idx = select_positions_by_key(F, r_low, r_high, key, min_mag_percentile=min_mag_percentile)
    if idx.shape[0] < len(stream_rep):
        return cover_rgb.copy(), np.empty((0, 2), dtype=np.int64), (r_low, r_high)

    # 4. QIM parity embedding
    used = []
    k = 0
    for r, c in idx:
        if k >= len(stream_rep):
            break
        Fst[int(r), int(c)] = _set_parity_logmag(Fst[int(r), int(c)], stream_rep[k], step)
        rr, cc = (M - int(r)) % M, (N - int(c)) % N
        if (rr, cc) != (int(r), int(c)):
            Fst[rr, cc] = np.conj(Fst[int(r), int(c)])
        used.append((int(r), int(c)))
        k += 1

    # 5. IFFT to get stego image
    stego_blue = np.real(ifft2(ifftshift(Fst)))
    stego_blue = np.clip(stego_blue, 0, 255)

    out = cover_rgb.copy()
    out[:, :, 2] = stego_blue
    return out, np.array(used, dtype=np.int64), (r_low, r_high)

def extract_blind_fft_rgb(stego_rgb, key,
                          repeat_n=3, embed_ratio=0.30,
                          step=0.05, min_mag_percentile=30,
                          return_debug=False, flip_parity=False):
    blue = stego_rgb[:, :, 2].astype(np.float64)
    F = fftshift(fft2(blue))

    r_low, r_high = _map_embed_ratio_to_band(embed_ratio)
    idx = select_positions_by_key(F, r_low, r_high, key, min_mag_percentile=min_mag_percentile)
    if len(idx) == 0:
        return ("", False, {"reason": "no_positions"}) if return_debug else ("", False)

    bits_rep = []
    for r, c in idx:
        b = _get_parity_logmag(F[r, c], step)
        if flip_parity:
            b ^= 1
        bits_rep.append(b)

    bits = majority_vote(bits_rep, repeat_n)
    pos = find_sync(bits)
    debug = {"sync_pos": pos, "bits_total": len(bits), "used": len(idx)}
    if pos < 0:
        if return_debug:
            debug["reason"] = "no_sync"
            return ("", False, debug)
        return ("", False)

    i = pos + len(SYNC)
    if i + HEADER_BITS > len(bits):
        if return_debug:
            debug["reason"] = "short_header"
            return ("", False, debug)
        return ("", False)

    L = _bits_to_int(bits[i:i + HEADER_BITS])  # bytes
    i += HEADER_BITS
    need = L * 8
    if i + need > len(bits):
        if return_debug:
            debug["reason"] = "short_payload"
            return ("", False, debug)
        return ("", False)

    payload = bits[i:i + need]
    text = bits_to_text_utf8(payload)
    # rough sync confidence: distance of SYNC around pos
    debug["sync_score"] = 100
    return (text, True, debug) if return_debug else (text, True)

