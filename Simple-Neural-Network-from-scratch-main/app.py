import pickle
from pathlib import Path
from typing import Any
import numpy as np
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "./model/multilayer_model.pkl"

app = Flask(__name__)

_MODEL_CACHE: dict[str, Any] | None = None

def _load_model() -> dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    required = ["W1", "b1", "W2", "b2"]
    for k in required:
        if k not in model:
            raise KeyError(f"model.pkl missing key: {k}")

    _MODEL_CACHE = model
    return model

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

def predict_proba(pixels_784: np.ndarray) -> np.ndarray:
    m = _load_model()
    W1: np.ndarray = m["W1"]
    b1: np.ndarray = m["b1"]
    W2: np.ndarray = m["W2"]
    b2: np.ndarray = m["b2"]

    x = pixels_784.astype(np.float32).reshape(784, 1)
    z1 = W1.dot(x) + b1
    a1 = _relu(z1)
    z2 = W2.dot(a1) + b2
    a2 = _softmax(z2)
    return a2[:, 0]

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    pixels = payload.get("pixels")
    if not isinstance(pixels, list) or len(pixels) != 784:
        return jsonify({"error": "Expected JSON: { pixels: number[784] }"}), 400

    try:
        arr = np.array(pixels, dtype=np.float32)
    except Exception:
        return jsonify({"error": "Invalid pixels array"}), 400

    proba = predict_proba(arr)
    pred = int(np.argmax(proba))

    return jsonify(
        {
            "pred": pred,
            "proba": [float(x) for x in proba.tolist()],
        }
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)