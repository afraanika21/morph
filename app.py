from flask import Flask
from config import Config
from routes.images import bp as images_bp
from routes.steg_fft import bp_steg_fft
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # ensure dirs exist
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["RESULT_DIR"], exist_ok=True)

    # blueprints
    app.register_blueprint(images_bp)
    app.register_blueprint(bp_steg_fft)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
