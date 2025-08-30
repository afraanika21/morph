# routes/images.py
from flask import Blueprint, render_template, request, jsonify, send_file, url_for, current_app
from PIL import Image
import os, io, uuid
from dsp.grayscale import blend_grayscale
from dsp.tonemap import apply_brightness_contrast
from utils.imaging import allowed, pil_to_bytes, to_numpy_rgb, to_pil
from dsp.conv import apply_gaussian_blur
from dsp.pipeline import apply_pipeline 
from dsp.vignette import apply_vignette
from dsp.edges import apply_edges
from dsp.freq import apply_lowpass_fft, apply_highpass_fft

bp = Blueprint("images", __name__)

@bp.get("/")
def index():
    return render_template("index.html")

@bp.post("/upload")
def upload():
    if "image" not in request.files:
        return "No file part", 400
    f = request.files["image"]
    if f.filename == "" or not allowed(f.filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid file", 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    file_id = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(current_app.config["UPLOAD_DIR"], file_id)
    f.save(save_path)

    return jsonify({
        "ok": True,
        "filename": file_id,
        "url": url_for("static", filename=f"uploads/{file_id}")
    })

# -------- APPLY ENDPOINTS --------

@bp.post("/apply/grayscale")
def apply_grayscale():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    strength_pct = float(data.get("strength", 100))
    strength = max(0.0, min(100.0, strength_pct)) / 100.0

    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = blend_grayscale(np_rgb, strength)
    out_pil = to_pil(out_np)

    return send_file(pil_to_bytes(out_pil, fmt="PNG"),
                     mimetype="image/png",
                     as_attachment=False,
                     download_name="preview.png")

@bp.post("/apply/brightness")
def apply_brightness():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    brightness_pct = float(data.get("brightness", 0.0))  # [-100..100]
    brightness_pct = max(-100.0, min(100.0, brightness_pct))

    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_brightness_contrast(np_rgb, brightness_pct=brightness_pct, contrast_pct=0.0)
    out_pil = to_pil(out_np)

    return send_file(pil_to_bytes(out_pil, fmt="PNG"),
                     mimetype="image/png",
                     as_attachment=False,
                     download_name="preview.png")

@bp.post("/apply/contrast")
def apply_contrast():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    contrast_pct = float(data.get("contrast", 0.0))  # [-100..100]
    contrast_pct = max(-100.0, min(100.0, contrast_pct))

    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_brightness_contrast(np_rgb, brightness_pct=0.0, contrast_pct=contrast_pct)
    out_pil = to_pil(out_np)

    return send_file(pil_to_bytes(out_pil, fmt="PNG"),
                     mimetype="image/png",
                     as_attachment=False,
                     download_name="preview.png")

@bp.post("/apply/lowpass")
def apply_lowpass():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    cutoff = float(data.get("cutoff", 0.15))  # 0..0.5
    cutoff = max(0.01, min(0.5, cutoff))
    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400
    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path): return "File not found", 404
    pil_img = Image.open(src_path)
    # (optional) downscale for preview if you added _downscale_for_preview
    # pil_img = _downscale_for_preview(pil_img, 1280)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_lowpass_fft(np_rgb, cutoff=cutoff, pad=True)
    return send_file(pil_to_bytes(to_pil(out_np), fmt="PNG"), mimetype="image/png",
                     as_attachment=False, download_name="preview.png")

@bp.post("/apply/highpass")
def apply_highpass():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    cutoff = float(data.get("cutoff", 0.15))     # 0..0.5
    highboost = float(data.get("highboost", 0.0))# 0=plain HPF, >0 = sharpen
    cutoff = max(0.01, min(0.5, cutoff)); highboost = max(0.0, min(5.0, highboost))
    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400
    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path): return "File not found", 404
    pil_img = Image.open(src_path)
    # pil_img = _downscale_for_preview(pil_img, 1280)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_highpass_fft(np_rgb, cutoff=cutoff, highboost=highboost, pad=True)
    return send_file(pil_to_bytes(to_pil(out_np), fmt="PNG"), mimetype="image/png",
                     as_attachment=False, download_name="preview.png")

@bp.post("/apply/blur")
def apply_blur():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    sigma = float(data.get("sigma", 0.0))  # 0..5
    sigma = max(0.0, min(5.0, sigma))

    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_gaussian_blur(np_rgb, sigma)
    out_pil = to_pil(out_np)

    return send_file(
        pil_to_bytes(out_pil, fmt="PNG"),
        mimetype="image/png",
        as_attachment=False,
        download_name="preview.png"
    )

@bp.post("/apply/pipeline")
def apply_pipeline_route():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)

    out_np = apply_pipeline(np_rgb, {
        "grayscale_strength": data.get("grayscale_strength"),
        "blur_sigma": data.get("blur_sigma"),
        "brightness": data.get("brightness"),
        "contrast": data.get("contrast"),
        "vignette_strength": data.get("vignette_strength"),  # <-- ADD THIS
    })
    out_pil = to_pil(out_np)

    return send_file(pil_to_bytes(out_pil, fmt="PNG"),
                     mimetype="image/png",
                     as_attachment=False,
                     download_name="preview.png")


@bp.post("/apply/vignette")
def apply_vignette_route():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    strength = float(data.get("strength", 0.5))
    strength = max(0.0, min(1.0, strength))

    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_vignette(np_rgb, strength)
    out_pil = to_pil(out_np)

    return send_file(pil_to_bytes(out_pil, fmt="PNG"),
                     mimetype="image/png",
                     as_attachment=False,
                     download_name="preview.png")


@bp.post("/apply/edges")
def apply_edges_route():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    kind = (data.get("kind") or "sobel").lower()
    threshold = float(data.get("threshold", 100.0))      # 0..255
    overlay = bool(data.get("overlay", False))
    overlay_alpha = float(data.get("overlay_alpha", 0.6))# 0..1
    overlay_color = data.get("overlay_color", [255,64,64])
    smooth_sigma = float(data.get("smooth_sigma", 1.0))
    return_map = bool(data.get("return_map", False))

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)
    out_np = apply_edges(np_rgb, kind=kind, threshold=threshold,
                         overlay=overlay, overlay_alpha=overlay_alpha,
                         overlay_color=overlay_color, smooth_sigma=smooth_sigma,
                         return_map=return_map)
    out_pil = to_pil(out_np)

    return send_file(pil_to_bytes(out_pil, fmt="PNG"),
                     mimetype="image/png",
                     as_attachment=False, download_name="preview.png")

# -------- SAVE (generic) --------


@bp.post("/save")
def save_result():
    """
    Also supports:
      op: "pipeline"
      with optional grayscale_strength, brightness, contrast
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    op = (data.get("op") or "").lower().strip()

    if not filename or not allowed(filename, current_app.config["ALLOWED_EXTS"]):
        return "Invalid filename", 400

    src_path = os.path.join(current_app.config["UPLOAD_DIR"], filename)
    if not os.path.exists(src_path):
        return "File not found", 404

    pil_img = Image.open(src_path)
    np_rgb = to_numpy_rgb(pil_img)

    if op == "grayscale":
        strength_pct = float(data.get("strength", 100.0))
        strength = max(0.0, min(100.0, strength_pct)) / 100.0
        out_np = blend_grayscale(np_rgb, strength)
        suffix = f"grayscale_{int(strength_pct)}"

    elif op == "vignette":  # <-- OPTIONAL single-op save
        vs = float(data.get("strength", 0.5))
        vs = max(0.0, min(1.0, vs))
        out_np = apply_vignette(np_rgb, vs)
        suffix = f"vignette_{vs:.2f}"
    
    elif op == "blur":
        sigma = max(0.0, min(5.0, float(data.get("sigma", 0.0))))
        out_np = apply_gaussian_blur(np_rgb, sigma)
        suffix = f"blur_s{sigma:.1f}"

    elif op == "brightness":
        brightness_pct = float(data.get("brightness", 0.0))
        brightness_pct = max(-100.0, min(100.0, brightness_pct))
        out_np = apply_brightness_contrast(np_rgb, brightness_pct=brightness_pct, contrast_pct=0.0)
        suffix = f"brightness_{int(brightness_pct)}"

    elif op == "contrast":
        contrast_pct = float(data.get("contrast", 0.0))
        contrast_pct = max(-100.0, min(100.0, contrast_pct))
        out_np = apply_brightness_contrast(np_rgb, brightness_pct=0.0, contrast_pct=contrast_pct)
        suffix = f"contrast_{int(contrast_pct)}"

    elif op == "lowpass":
        cutoff = max(0.01, min(0.5, float(data.get("cutoff", 0.15))))
        out_np = apply_lowpass_fft(np_rgb, cutoff=cutoff, pad=True)
        suffix = f"lpf_c{cutoff:.2f}"

    elif op == "highpass":
        cutoff = max(0.01, min(0.5, float(data.get("cutoff", 0.15))))
        highboost = max(0.0, min(5.0, float(data.get("highboost", 0.0))))
        out_np = apply_highpass_fft(np_rgb, cutoff=cutoff, highboost=highboost, pad=True)
        suffix = f"hpf_c{cutoff:.2f}_hb{highboost:.2f}"

    elif op == "edges":
        kind = (data.get("kind") or "sobel").lower()
        threshold = float(data.get("threshold", 100.0))
        overlay = bool(data.get("overlay", False))
        overlay_alpha = float(data.get("overlay_alpha", 0.6))
        overlay_color = data.get("overlay_color", [255,64,64])
        smooth_sigma = float(data.get("smooth_sigma", 1.0))
        return_map = bool(data.get("return_map", False))

        out_np = apply_edges(np_rgb, kind=kind, threshold=threshold,
                             overlay=overlay, overlay_alpha=overlay_alpha,
                             overlay_color=overlay_color, smooth_sigma=smooth_sigma,
                             return_map=return_map)
        suffix = f"edges_{kind}_t{int(threshold)}"
        if overlay: suffix += f"_ov{overlay_alpha:.2f}"

    elif op == "pipeline":
        gs = data.get("grayscale_strength", None)
        br = data.get("brightness", None)
        ct = data.get("contrast", None)
        bs = data.get("blur_sigma", None)  # <-- NEW
        vs = data.get("vignette_strength", None)  # <-- ADD
        out_np = apply_pipeline(np_rgb, {
            "grayscale_strength": gs,
            "blur_sigma": bs,
            "brightness": br,
            "contrast": ct,
            "vignette_strength": vs,  # <-- ADD
        })
        parts = []
        if gs is not None: parts.append(f"gs{int(float(gs))}")
        if bs is not None: parts.append(f"s{float(bs):.1f}")
        if br is not None: parts.append(f"b{int(float(br))}")
        if ct is not None: parts.append(f"c{int(float(ct))}")
        if vs is not None: parts.append(f"v{float(vs):.2f}")  # <-- ADD
        suffix = "pipeline_" + ("_".join(parts) if parts else "noop")

    else:
        return "Unknown op", 400

    out_pil = to_pil(out_np)
    out_name = f"{suffix}_{uuid.uuid4().hex}.png"
    out_path = os.path.join(current_app.config["RESULT_DIR"], out_name)
    out_pil.save(out_path, format="PNG")

    return jsonify({
        "ok": True,
        "result_url": url_for("static", filename=f"results/{out_name}", _external=False)
    })
