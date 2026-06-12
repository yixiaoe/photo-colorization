"""
Photo Colorization Demo — Flask entry point.

Run from code/ directory:
    python app.py

Then open:
    http://localhost:5000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web'))

from flask import Flask, send_from_directory
from web.api.detect import detect_bp
from web.api.colorize import colorize_bp

app = Flask(__name__, static_folder='web', static_url_path='')


@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


app.register_blueprint(detect_bp)
app.register_blueprint(colorize_bp)


if __name__ == '__main__':
    print('[app] Photo Colorization Demo → http://localhost:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)
