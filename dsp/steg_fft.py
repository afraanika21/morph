import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ---------------- Bit helpers ---------------- #

def _int_to_bits(x: int, width: int):
    return [int(b) for b in format(x, f"0{width}b")]

def _bits_to_int(bits):
    return int("".join(map(str, bits)), 2) if bits else 0

def text_to_bits_utf8(text: str):
    b = text.encode("utf-8")
    return [int(bx) for byte in b for bx in format(byte, "08b")]

def bits_to_text_utf8(bits):
    if not bits:
        return ""
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    out = bytearray()
    for i in range(0, n, 8):
        out.append(_bits_to_int(bits[i:i+8]))
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return "[decode error]"

def repeat_bits(bits, n=3):
    return [bit for bit in bits for _ in range(n)]

def majority_vote(bits, n=3):
    if n <= 1:
        return bits[:]
    out = []
    for i in range(0, len(bits), n):
        grp = bits[i:i+n]
        out.append(1 if sum(grp) > (len(grp)//2) else 0)
    return out

# ---------------- Framing ---------------- #

HEADER_BITS = 32
REPEAT_N = 3

def build_bitstream(text: str, repeat_n: int = REPEAT_N):
    payload_bits = text_to_bits_utf8(text)
    length_bytes = len(payload_bits) // 8
    header_bits = _int_to_bits(length_bytes, HEADER_BITS)
    stream = header_bits + payload_bits
    return repeat_bits(stream, n=repeat_n), length_bytes

def parse_bitstream(bits_repeated, repeat_n: int = REPEAT_N):
    collapsed = majority_vote(bits_repeated, n=repeat_n)
    if len(collapsed) < HEADER_BITS:
        return [], 0, False

    header = collapsed[:HEADER_BITS]
    msg_len_bytes = _bits_to_int(header)
    payload_bits_needed = msg_len_bytes * 8
    if len(collapsed) - HEADER_BITS < payload_bits_needed:
        return [], msg_len_bytes, False

    payload_bits = collapsed[HEADER_BITS:HEADER_BITS+payload_bits_needed]
    return payload_bits, msg_len_bytes, True

# ---------------- FFT helpers ---------------- #

def _annulus_mask(M, N, r_low=0.45, r_high=0.85):
    cx, cy = M // 2, N // 2
    Y, X = np.ogrid[:M, :N]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    inner, outer = maxd*r_low, maxd*r_high
    return (dist >= inner) & (dist <= outer)

def _select_positions(F, mask, min_mag_percentile=30):
    mag = np.abs(F[mask])
    if mag.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    thr = np.percentile(mag, min_mag_percentile)
    R, C = np.where(mask)
    keep = (np.abs(F[R, C]) >= thr)
    idx = np.column_stack((R[keep], C[keep]))
    if idx.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(42)
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
    blue = cover_rgb[:, :, 2].astype(np.float64, copy=True)
    M, N = blue.shape

    F = fftshift(fft2(blue))
    F_stego = F.copy()

    mask = _annulus_mask(M, N, r_low, r_high)
    idx = _select_positions(F, mask, min_mag_percentile)

    if len(idx) < len(message_bits_repeated):
        return cover_rgb.copy(), np.empty((0, 2), dtype=np.int64)

    used = []
    count = 0
    for r, c in idx:
        if count >= len(message_bits_repeated):
            break
        val = F_stego[r, c]
        mag = np.abs(val)
        if mag < 1e-12:
            continue
        phase = np.angle(val)
        bit = message_bits_repeated[count]
        new_mag = mag*(1+strength_factor) if bit else mag*(1-strength_factor)
        F_stego[r, c] = new_mag*np.exp(1j*phase)
        rr, cc = (M-r) % M, (N-c) % N
        if (rr, cc) != (r, c):
            F_stego[rr, cc] = np.conj(F_stego[r, c])
        used.append((r, c))
        count += 1

    stego_blue = np.real(ifft2(ifftshift(F_stego)))
    stego_blue = np.clip(stego_blue, 0, 255)
    out = cover_rgb.copy()
    out[:, :, 2] = stego_blue
    return out, np.array(used, dtype=np.int64)

def extract_message_fft_rgb(stego_rgb, original_rgb, used_positions, strength_factor=0.10):
    stegoF = fftshift(fft2(stego_rgb[:, :, 2].astype(np.float64)))
    origF = fftshift(fft2(original_rgb[:, :, 2].astype(np.float64)))
    out_bits = []
    for r, c in used_positions:
        ms, mo = np.abs(stegoF[r, c]), np.abs(origF[r, c])
        if mo < 1e-12:
            out_bits.append(1 if ms > 1e-12 else 0)
            continue
        d1 = abs(ms - mo*(1+strength_factor))
        d0 = abs(ms - mo*(1-strength_factor))
        out_bits.append(1 if d1 < d0 else 0)
    return out_bits

# ---------------- Capacity estimate ---------------- #

def estimate_capacity_bits(shape_hw, r_low=0.45, r_high=0.85):
    M, N = shape_hw
    cx, cy = M//2, N//2
    maxd = np.sqrt(cx**2 + cy**2)
    area_annulus = np.pi*((r_high*maxd)**2 - (r_low*maxd)**2)
    return int(area_annulus)
