import io
import numpy as np
from PIL import Image

def allowed(filename: str, allowed_exts: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts

def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> io.BytesIO:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def to_numpy_rgb(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.float32)

def to_pil(np_img: np.ndarray) -> Image.Image:
    np_clip = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_clip, mode="RGB")
