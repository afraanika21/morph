# routes/blind_steg_route.py
import os, uuid, json, traceback, inspect
from flask import Blueprint, request, jsonify, current_app, url_for, render_template
from PIL import Image
import numpy as np

from utils.imaging import to_numpy_rgb, to_pil, allowed
from dsp import blind_steg_fft as bs  # module import so we can debug __file__

bp_blind = Blueprint("blind_steg", __name__)

@bp_blind.get("/blind")
def blind_index():
    return render_template("blind_steg.html")

# --- Embed (blind) ---
@bp_blind.post("/blind/embed")
def blind_embed():
    try:
        # --- Standard Blind Steganography Simulation ---
        data        = request.get_json(silent=True) or {}
        filename    = data.get("filename")
        text        = data.get("text", "")
        key         = (data.get("key") or "").strip()
        repeat_n    = int(float(data.get("repeat_n", 3)))
        embed_ratio = float(data.get("embed_ratio", 0.30))
        step        = float(data.get("step", 0.05))
        minp        = int(float(data.get("min_mag_percentile", 30)))

        # Validate inputs
        if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
            return jsonify({"ok": False, "error": "Invalid file"}), 400
        if not key:
            return jsonify({"ok": False, "error": "Missing key"}), 400

        src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
        if not os.path.exists(src_path):
            return jsonify({"ok": False, "error": "File not found"}), 404

        # Load cover image
        img = Image.open(src_path).convert("RGB")
        np_rgb = to_numpy_rgb(img).astype(np.float64)

        # DSP: FFT, key-based position selection, QIM, repetition coding
        stego_rgb, used, (r_low, r_high) = bs.embed_blind_fft_rgb(
            np_rgb, text, key,
            repeat_n=repeat_n,
            embed_ratio=embed_ratio,
            step=step,
            min_mag_percentile=minp
        )
        if used is None or len(used) == 0:
            return jsonify({"ok": False, "error": "Not enough capacity"}), 400

        # Save stego image
        out_pil = to_pil(stego_rgb)
        out_name = f"blind_{uuid.uuid4().hex}.png"
        out_path = os.path.join(current_app.config["RESULT_DIR"], out_name)
        out_pil.save(out_path)

        # Save sidecar meta for reproducibility
        meta = {
            "repeat_n": repeat_n,
            "embed_ratio": embed_ratio,
            "step": step,
            "min_mag_percentile": minp,
            "r_low": r_low,
            "r_high": r_high
        }
        try:
            with open(out_path.replace(".png", ".json"), "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

        # Output stego URL for download and extraction
        return jsonify({"ok": True,
                        "steg_url": url_for("static", filename=f"results/{out_name}")})
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"ok": False, "error": f"Embed failed:\n{tb}"}), 500

# --- Extract (blind) ---
@bp_blind.post("/blind/extract")
def blind_extract():
    try:
        data          = request.get_json(silent=True) or {}
        steg_filename = (data.get("steg_filename") or "").strip()
        steg_source   = (data.get("steg_source") or "results").lower()
        key           = (data.get("key") or "").strip()
        repeat_n      = int(float(data.get("repeat_n", 3)))
        embed_ratio   = float(data.get("embed_ratio", 0.50))
        step          = float(data.get("step", 0.12))
        minp          = int(float(data.get("min_mag_percentile", 30)))

        if not steg_filename:
            return jsonify({"ok": False, "error": "Missing stego filename"}), 400
        if not key:
            return jsonify({"ok": False, "error": "Missing key"}), 400

        # Normalize filename
        steg_filename = os.path.basename(steg_filename).split("?", 1)[0].split("#", 1)[0]

        uploads_dir = current_app.config["UPLOAD_DIR"]
        results_dir = current_app.config["RESULT_DIR"]
        primary_dir = results_dir if steg_source != "uploads" else uploads_dir
        alt_dir     = uploads_dir if primary_dir == results_dir else results_dir

        candidates = [
            os.path.join(primary_dir, steg_filename),
            os.path.join(alt_dir, steg_filename),
        ]
        steg_path = next((p for p in candidates if os.path.exists(p)), None)
        if not steg_path:
            return jsonify({
                "ok": False,
                "error": "Stego file not found.\nTried:\n- " + "\n- ".join(candidates)
            }), 404

        # Load stego image
        stego = to_numpy_rgb(Image.open(steg_path).convert("RGB")).astype(np.float64)

        # Meta sidecar
        meta_path = steg_path.replace(".png", ".json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                repeat_n    = int(meta.get("repeat_n", repeat_n))
                embed_ratio = float(meta.get("embed_ratio", embed_ratio))
                step        = float(meta.get("step", step))
                minp        = int(meta.get("min_mag_percentile", minp))
            except Exception:
                pass

        # --- Try extraction ---
        best = {"score": -1, "ok": False, "text": "", "params": {}}

        for rep in [repeat_n, max(1, repeat_n-1), repeat_n+1]:
            for st in [step, max(0.02, step*0.85), min(0.25, step*1.15)]:
                for er in [embed_ratio,
                           max(0.10, embed_ratio-0.10),
                           min(0.90, embed_ratio+0.10)]:
                    for mp in [minp, max(5, minp-10), min(90, minp+10)]:
                        for flip in (False, True):
                            text, ok, info = bs.extract_blind_fft_rgb(
                                stego, key,
                                repeat_n=rep, embed_ratio=er, step=st, min_mag_percentile=mp,
                                return_debug=True, flip_parity=flip
                            )
                            score = (1 if ok else 0) * 100000 + len(text)
                            if isinstance(info, dict) and "sync_score" in info:
                                score += int(info["sync_score"])
                            if score > best["score"]:
                                best = {
                                    "score": score,
                                    "ok": ok,
                                    "text": text,
                                    "params": {
                                        "repeat_n": int(rep),
                                        "embed_ratio": float(er),
                                        "step": float(st),
                                        "min_mag_percentile": int(mp),
                                        "flip_parity": bool(flip),
                                    }
                                }
                            if ok:
                                return jsonify({"ok": True,
                                                "message": text,
                                                "used_params": best["params"]})

        return jsonify({
            "ok": False,
            "message": best["text"],
            "error": "Blind decode failed. Try adjusting step (0.14–0.18), band (0.5–0.6), repetition (3–4), and key.",
            "best_params_tried": best["params"]
        })
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


