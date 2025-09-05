# routes/noise.py
from flask import Blueprint, render_template, request, jsonify, send_file, url_for, current_app
from PIL import Image
import io, os, uuid, numpy as np

from utils.imaging import to_numpy_rgb, to_pil, allowed
from dsp.noise import add_awgn, add_salt_pepper, add_speckle, add_poisson
from dsp.denoise import moving_average, median_denoise, psnr as psnr_fn, ssim as ssim_fn

bp_noise = Blueprint("noise", __name__)

@bp_noise.get("/noise")
def noise_index():
    return render_template("noise.html")

def _load_rgb_from_uploads(filename):
    path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not (filename and allowed(filename, current_app.config["ALLOWED_EXTS"]) and os.path.exists(path)):
        return None, None
    pil = Image.open(path).convert("RGB")
    return to_numpy_rgb(pil), path

def _load_rgb_from_results(filename):
    path = os.path.join(current_app.config["RESULT_DIR"], filename)
    if not (filename and os.path.exists(path)):
        return None, None
    pil = Image.open(path).convert("RGB")
    return to_numpy_rgb(pil), path

def _send_png(np_rgb):
    pil = to_pil(np_rgb)
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    bio.seek(0)
    return send_file(bio, mimetype="image/png", as_attachment=False, download_name="preview.png")

# ---------------- Add Noise ---------------- #

@bp_noise.post("/noise/add")
def noise_add():
    """
    Preview: returns PNG blob
    Commit:  add ?commit=1 -> returns JSON {ok, noisy_url, noisy_filename}
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    ntype = (data.get("type") or "awgn").lower()

    # params
    sigma = float(data.get("sigma", 20.0))
    seed  = data.get("seed", None)
    sp_p  = float(data.get("p", 0.02))
    speckle_sigma = float(data.get("speckle_sigma", 0.15))
    pois_scale    = float(data.get("poisson_scale", 12.0))

    commit = request.args.get("commit", "0") == "1"

    src_np, _ = _load_rgb_from_uploads(filename)
    if src_np is None:
        return "Invalid or missing source file", 400

    if ntype == "awgn":
        out = add_awgn(src_np, sigma=sigma, seed=seed)
        tag = f"awgn_s{sigma:.0f}"
    elif ntype == "sp":
        out = add_salt_pepper(src_np, p=sp_p, seed=seed)
        tag = f"sp_p{sp_p:.2f}"
    elif ntype == "speckle":
        out = add_speckle(src_np, sigma=speckle_sigma, seed=seed)
        tag = f"speckle_s{speckle_sigma:.2f}"
    elif ntype == "poisson":
        out = add_poisson(src_np, scale=pois_scale, seed=seed)
        tag = f"poisson_{pois_scale:.0f}"
    else:
        return "Unknown noise type", 400

    if not commit:
        return _send_png(out)

    # Commit -> save to results
    out_pil = to_pil(out)
    out_name = f"noisy_{tag}_{uuid.uuid4().hex}.png"
    out_path = os.path.join(current_app.config["RESULT_DIR"], out_name)
    out_pil.save(out_path, format="PNG")
    return jsonify({
        "ok": True,
        "noisy_url": url_for("static", filename=f"results/{out_name}"),
        "noisy_filename": out_name
    })

# ---------------- Denoise ---------------- #

@bp_noise.post("/noise/denoise")
def noise_denoise():
    """
    Source: noisy image saved in results (filename),
    Preview (PNG) vs Commit (?commit=1 -> JSON).
    """
    data = request.get_json(silent=True) or {}
    source_filename = data.get("source_filename")  # must be a saved noisy file in results
    method = (data.get("method") or "ma").lower()

    commit = request.args.get("commit", "0") == "1"

    src_np, _ = _load_rgb_from_results(source_filename)
    if src_np is None:
        return "Invalid or missing source noisy file", 400

    if method == "ma":
        k = int(float(data.get("k", 3)))
        out = moving_average(src_np, k=k)
        tag = f"ma_k{k}"
    elif method == "median":
        k = int(float(data.get("k", 3)))
        out = median_denoise(src_np, k=k)
        tag = f"median_k{k}"
    else:
        return "Unknown denoise method", 400

    if not commit:
        return _send_png(out)

    out_pil = to_pil(out)
    out_name = f"denoised_{tag}_{uuid.uuid4().hex}.png"
    out_path = os.path.join(current_app.config["RESULT_DIR"], out_name)
    out_pil.save(out_path, format="PNG")
    return jsonify({
        "ok": True,
        "denoised_url": url_for("static", filename=f"results/{out_name}"),
        "denoised_filename": out_name
    })

# ---------------- Metrics ---------------- #

@bp_noise.post("/noise/metrics")
def noise_metrics():
    """
    Compute PSNR/SSIM between any two saved images (uploads/results).
    Payload:
      { "ref": {"where":"uploads|results", "filename":"..."},
        "test":{"where":"uploads|results", "filename":"..."} }
    """
    data = request.get_json(silent=True) or {}

    def load(where, filename):
        if where == "uploads":
            return _load_rgb_from_uploads(filename)[0]
        else:
            return _load_rgb_from_results(filename)[0]

    ref_spec  = data.get("ref", {})
    test_spec = data.get("test", {})

    ref = load(ref_spec.get("where","uploads"), ref_spec.get("filename"))
    tst = load(test_spec.get("where","results"), test_spec.get("filename"))
    if ref is None or tst is None:
        return jsonify({"ok": False, "error": "Files not found"}), 400

    p = psnr_fn(ref, tst)
    s = ssim_fn(ref, tst)
    return jsonify({"ok": True, "psnr": float(p), "ssim": float(s)})
