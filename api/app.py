import os

from flask import Flask, send_from_directory

from api.routes import api_bp

# Served from the same origin as /api, so the browser needs no CORS or dev proxy.
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

app.register_blueprint(api_bp, url_prefix='/api')


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
