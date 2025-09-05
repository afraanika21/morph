import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ---------------- Bit helpers ---------------- #

def _int_to_bits(x: int, width: int):
    """Return 'width' bits (MSB first) for integer x."""
    s = format(int(x), f"0{int(width)}b")
    return [int(ch) for ch in s]

def _bits_to_int(bits):
    """Convert a list of 0/1 into an integer (MSB first)."""
    if not bits:
        return 0
    return int("".join("1" if b else "0" for b in bits), 2)

def text_to_bits_utf8(text: str):
    """Convert UTF-8 text to a flat list of bits (MSB first per byte)."""
    data = text.encode("utf-8")
    return [int(bit) for byte in data for bit in f"{byte:08b}"]

def bits_to_text_utf8(bits):
    if not bits:
        return ""
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    out = bytearray()
    for i in range(0, n, 8):
        byte = int("".join(str(b) for b in bits[i:i+8]), 2)   # ✅ here b is the loop variable
        out.append(byte)
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return "[decode error]"


def repeat_bits(bits, n=3):
    """Repeat each bit n times for simple repetition coding."""
    n = max(1, int(n))
    return [bit for bit in bits for _ in range(n)]

def majority_vote(bits, n=3):
    """
    Collapse a repeated-bit stream by majority vote over groups of size n.
    If the final group is short, we still vote on it.
    """
    n = max(1, int(n))
    if n == 1:
        return bits[:]
    out = []
    for i in range(0, len(bits), n):
        grp = bits[i:i+n]
        out.append(1 if sum(grp) >= (len(grp) / 2.0) else 0)
    return out

# ---------------- Framing ---------------- #

HEADER_BITS = 32      # length header (bytes) as 32-bit unsigned
REPEAT_N = 3          # default repetition

def build_bitstream(text: str, repeat_n: int = REPEAT_N):
    """
    Build 'header + payload' and then apply repetition coding.
    Header = 32-bit payload length in BYTES (not chars).
    Returns (repeated_bits, length_bytes)
    """
    payload_bits = text_to_bits_utf8(text)
    length_bytes = len(payload_bits) // 8
    header_bits = _int_to_bits(length_bytes, HEADER_BITS)
    stream = header_bits + payload_bits
    return repeat_bits(stream, n=repeat_n), length_bytes

def parse_bitstream(bits_repeated, repeat_n: int = REPEAT_N):
    """
    Inverse of build_bitstream given the repeated-bit stream.
    Returns (payload_bits, msg_len_bytes, ok)
    """
    collapsed = majority_vote(bits_repeated, n=repeat_n)
    if len(collapsed) < HEADER_BITS:
        return [], 0, False

    header = collapsed[:HEADER_BITS]
    msg_len_bytes = _bits_to_int(header)
    need = int(msg_len_bytes) * 8

    if len(collapsed) - HEADER_BITS < need:
        return [], msg_len_bytes, False

    payload_bits = collapsed[HEADER_BITS:HEADER_BITS + need]
    return payload_bits, msg_len_bytes, True

# ---------------- FFT helpers ---------------- #

def _annulus_mask(M, N, r_low=0.45, r_high=0.85):
    """
    Boolean mask for a mid-band annulus in the 2D spectrum.
    r_* are fractions of the max radius.
    """
    cx, cy = M // 2, N // 2
    Y, X = np.ogrid[:M, :N]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    inner, outer = maxd * float(r_low), maxd * float(r_high)
    return (dist >= inner) & (dist <= outer)

def _select_positions(F, mask, min_mag_percentile=30):
    """
    Select coefficient positions within 'mask' above a magnitude percentile,
    then shuffle deterministically (seeded) for stable previews/tests.
    """
    R, C = np.where(mask)
    if R.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    mags = np.abs(F[R, C])
    if mags.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    thr = np.percentile(mags, float(min_mag_percentile))
    keep = mags >= max(thr, 1e-12)
    R, C = R[keep], C[keep]
    if R.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    idx = np.column_stack((R, C))
    rng = np.random.default_rng(42)   # deterministic shuffle for demo
    rng.shuffle(idx)
    return idx

# ---------------- Embed / Extract ---------------- #

def embed_message_fft_rgb(
    cover_rgb,
    message_bits_repeated,
    strength_factor=0.10,
    r_low=0.45,
    r_high=0.85,
    min_mag_percentile=30
):
    """
    NON-BLIND spectral stego (needs 'used_positions' to extract).
    Multiplies selected FFT magnitudes by (1±strength_factor) based on bit.
    Returns (stego_rgb, used_positions)
    """
    blue = cover_rgb[:, :, 2].astype(np.float64, copy=True)
    M, N = blue.shape

    F = fftshift(fft2(blue))
    F_stego = F.copy()

    mask = _annulus_mask(M, N, r_low, r_high)
    idx = _select_positions(F, mask, min_mag_percentile)

    total_bits = len(message_bits_repeated)
    if idx.shape[0] < total_bits:
        # Not enough positions to carry the repeated message.
        return cover_rgb.copy(), np.empty((0, 2), dtype=np.int64)

    used = []
    k = 0
    for r, c in idx:
        if k >= total_bits:
            break
        val = F_stego[r, c]
        mag = np.abs(val)
        if mag < 1e-12:
            continue
        phase = np.angle(val)
        bit = int(message_bits_repeated[k])
        # Apply multiplicative tweak
        new_mag = mag * (1.0 + float(strength_factor)) if bit else mag * (1.0 - float(strength_factor))
        F_stego[r, c] = new_mag * np.exp(1j * phase)

        # Conjugate-symmetric partner to keep output real
        rr, cc = (M - r) % M, (N - c) % N
        if (rr, cc) != (r, c):
            F_stego[rr, cc] = np.conj(F_stego[r, c])

        used.append((int(r), int(c)))
        k += 1

    # Synthesize the stego image
    stego_blue = np.real(ifft2(ifftshift(F_stego)))
    stego_blue = np.clip(stego_blue, 0, 255)

    out = cover_rgb.copy()
    out[:, :, 2] = stego_blue
    return out, np.asarray(used, dtype=np.int64)

def extract_message_fft_rgb(stego_rgb, original_rgb, used_positions, strength_factor=0.10):
    """
    NON-BLIND extraction (requires original or at least 'used_positions').
    Decides each bit by comparing stego/original magnitude against (1±strength).
    """
    if used_positions is None or len(used_positions) == 0:
        return []

    stego_blue = stego_rgb[:, :, 2].astype(np.float64)
    orig_blue  = original_rgb[:, :, 2].astype(np.float64)

    stegoF = fftshift(fft2(stego_blue))
    origF  = fftshift(fft2(orig_blue))

    out_bits = []
    s = float(strength_factor)
    eps = 1e-12
    for r, c in used_positions:
        ms = np.abs(stegoF[int(r), int(c)])
        mo = np.abs(origF[int(r), int(c)])
        if mo < eps:
            # Fallback if original magnitude is too small
            out_bits.append(1 if ms > eps else 0)
            continue
        # Compare distances to expected (1+s) and (1-s) scaling
        d1 = abs(ms - mo * (1.0 + s))
        d0 = abs(ms - mo * (1.0 - s))
        out_bits.append(1 if d1 < d0 else 0)
    return out_bits

# ---------------- Capacity estimate ---------------- #

def estimate_capacity_bits(shape_hw, r_low=0.45, r_high=0.85):
    """
    Very rough capacity estimate ~ area of annulus (unique bins).
    This ignores percentile filtering and repetition coding overhead.
    """
    M, N = map(int, shape_hw)
    cx, cy = M // 2, N // 2
    maxd = np.sqrt(cx**2 + cy**2)
    inner = float(r_low) * maxd
    outer = float(r_high) * maxd
    area_annulus = np.pi * (outer**2 - inner**2)
    # Approx. half the spectrum carries unique (non-conjugate) bins.
    return int(area_annulus / 2.0)
