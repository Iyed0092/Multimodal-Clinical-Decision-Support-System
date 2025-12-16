from flask import Flask
from flask_cors import CORS

from backend.mri import mri_bp
from backend.xray import xray_bp
from backend.nlp import nlp_bp


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.register_blueprint(mri_bp, url_prefix="/api/mri")
app.register_blueprint(xray_bp, url_prefix="/api/xray")
app.register_blueprint(nlp_bp, url_prefix="/api/nlp")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
