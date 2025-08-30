import os

class Config:
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    STATIC_DIR = os.path.join(APP_ROOT, "static")
    UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
    RESULT_DIR = os.path.join(STATIC_DIR, "results")

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTS = {"png", "jpg", "jpeg"}
