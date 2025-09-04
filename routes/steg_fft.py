from flask import Blueprint, request, jsonify, current_app, url_for, render_template
from PIL import Image
import os, uuid, json, numpy as np
from utils.imaging import to_numpy_rgb, to_pil, allowed
from dsp.steg_fft import (
    build_bitstream, parse_bitstream, bits_to_text_utf8,
    embed_message_fft_rgb, extract_message_fft_rgb
)

bp_steg_fft = Blueprint("steg_fft", __name__)

# -------- Page --------
@bp_steg_fft.get("/steg")
def steg_index():
    return render_template("steg_fft.html")

def _map_embed_ratio_to_band(embed_ratio: float):
    """
    Frontend provides a single 'embed_ratio' (0.1..0.9).
    We'll map it to a mid-band annulus [r_low, r_high].
    Strategy:
      - use the outer band cap ~0.88 (keep a little safety margin)
      - set r_low so thickness equals embed_ratio * (max radius)
    """
    embed_ratio = float(embed_ratio)
    r_high = 0.88
    thickness = max(0.05, min(0.7, embed_ratio))     # clamp
    r_low = max(0.05, r_high - thickness)
    return r_low, r_high

# -------- Embed --------
@bp_steg_fft.post("/stegfft/embed")
def stegfft_embed():
    try:
        data = request.get_json(silent=True) or {}
        filename  = data.get("filename")
        message   = data.get("text", "")

        # sliders (with defaults)
        strength   = float(data.get("strength", 0.10))      # 0.01..0.2 typical
        embed_ratio = float(data.get("embed_ratio", 0.30))  # 0.1..0.9
        repeat_n   = int(data.get("repeat_n", 3))           # 1..5

        if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
            return jsonify({"ok": False, "error": "Invalid file"}), 400

        src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
        if not os.path.exists(src_path):
            return jsonify({"ok": False, "error": "File not found"}), 404

        pil_img = Image.open(src_path).convert("RGB")
        np_rgb  = to_numpy_rgb(pil_img).astype(np.float64)

        # Build repeated bitstream with 32-bit length header
        bits_repeated, msg_len = build_bitstream(message, repeat_n=repeat_n)

        # Map single slider to annulus
        r_low, r_high = _map_embed_ratio_to_band(embed_ratio)

        # Embed
        stego_rgb, used = embed_message_fft_rgb(
            np_rgb,
            bits_repeated,
            strength_factor=strength,
            r_low=r_low, r_high=r_high,
            min_mag_percentile=30
        )

        if used.size == 0 or len(used) < len(bits_repeated):
            return jsonify({"ok": False, "error": "Message too long for cover image"}), 400

        # Save stego
        out_name = f"stegfft_{uuid.uuid4().hex}.png"
        out_path = os.path.join(current_app.config["RESULT_DIR"], out_name)
        to_pil(stego_rgb).save(out_path)

        # Persist positions + meta so extraction matches params
        np.save(out_path.replace(".png", "_used.npy"), used)
        meta = {
            "strength_factor": strength,
            "embed_ratio": embed_ratio,
            "repeat_n": repeat_n,
            "r_low": r_low, "r_high": r_high,
            "min_mag_percentile": 30
        }
        with open(out_path.replace(".png", ".json"), "w") as f:
            json.dump(meta, f)

        return jsonify({
            "ok": True,
            "steg_url": url_for("static", filename=f"results/{out_name}"),
            "original_url": url_for("static", filename=f"uploads/{filename}"),
            "len_bytes": msg_len
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -------- Extract --------
@bp_steg_fft.post("/stegfft/extract")
def stegfft_extract():
    try:
        data = request.get_json(silent=True) or {}
        orig_file = data.get("orig_filename")
        steg_file = data.get("steg_filename")

        if not orig_file or not steg_file:
            return jsonify({"ok": False, "error": "Need both filenames"}), 400

        orig_path = os.path.join(current_app.config["UPLOAD_DIR"],  orig_file)
        steg_path = os.path.join(current_app.config["RESULT_DIR"],  steg_file)
        used_path = steg_path.replace(".png", "_used.npy")
        meta_path = steg_path.replace(".png", ".json")

        if not (os.path.exists(orig_path) and os.path.exists(steg_path) and os.path.exists(used_path)):
            return jsonify({"ok": False, "error": "Files missing"}), 404

        # Default meta (in case JSON missing)
        meta = {
            "strength_factor": 0.10,
            "repeat_n": 3
        }
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta.update(json.load(f))
            except Exception:
                pass

        orig = to_numpy_rgb(Image.open(orig_path).convert("RGB")).astype(np.float64)
        steg = to_numpy_rgb(Image.open(steg_path).convert("RGB")).astype(np.float64)
        used = np.load(used_path, allow_pickle=True)

        # Extract with the same strength used for embedding
        bits_rep = extract_message_fft_rgb(
            steg, orig, used,
            strength_factor=float(meta.get("strength_factor", 0.10))
        )

        # Parse header + collapse repetition using saved repeat_n
        payload_bits, msg_len_bytes, ok = parse_bitstream(
            bits_rep,
            repeat_n=int(meta.get("repeat_n", 3))
        )
        text = bits_to_text_utf8(payload_bits)

        return jsonify({
            "ok": ok,
            "message": text,
            "len_bytes": msg_len_bytes
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
