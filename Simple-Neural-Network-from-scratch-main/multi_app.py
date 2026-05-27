import pickle
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "multilayer_model.pkl"

app = Flask(__name__)

_MODEL_CACHE: dict[str, Any] | None = None


class _ModelBundle(TypedDict):
    params: dict[str, np.ndarray]
    layer_dims: list[int]


def _load_model() -> _ModelBundle:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE  # type: ignore[return-value]

    with open(MODEL_PATH, "rb") as f:
        raw: Any = pickle.load(f)

    if isinstance(raw, dict) and "params" in raw:
        params: dict[str, np.ndarray] = raw["params"]
        layer_dims: list[int] | None = raw.get("layer_dims")
    elif isinstance(raw, dict) and all(k in raw for k in ("W1", "b1", "W2", "b2")):
        # Legacy 2-layer: W1 (h,784), W2 (out,h)
        params = {k: raw[k] for k in ("W1", "b1", "W2", "b2")}
        layer_dims = [
            int(raw["W1"].shape[1]),
            int(raw["W1"].shape[0]),
            int(raw["W2"].shape[0]),
        ]
    else:
        raise KeyError(
            "Model file must be either "
            "{'params': {...}, 'layer_dims': [...]} "
            "or flat keys W1,b1,W2,b2"
        )

    L = len(params) // 2
    for l in range(1, L + 1):
        for key in (f"W{l}", f"b{l}"):
            if key not in params:
                raise KeyError(f"params missing {key}")

    if layer_dims is not None and len(layer_dims) - 1 != L:
        raise ValueError(
            f"layer_dims length mismatch: got {len(layer_dims) - 1} layers "
            f"from layer_dims but {L} from params"
        )

    _MODEL_CACHE = {"params": params, "layer_dims": layer_dims or []}
    return _MODEL_CACHE  # type: ignore[return-value]


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)


def _forward_prop(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Giống foward_prop trong Multilayer.ipynb: ReLU giữa các lớp, softmax ở đầu ra."""
    A = X
    L = len(params) // 2
    for l in range(1, L):
        z = params[f"W{l}"].dot(A) + params[f"b{l}"]
        A = _relu(z)
    z_out = params[f"W{L}"].dot(A) + params[f"b{L}"]
    return _softmax(z_out)


def predict_proba(pixels_784: np.ndarray) -> np.ndarray:
    bundle = _load_model()
    params = bundle["params"]

    # Canvas gửi 0–255; training dùng /255.0
    x = (pixels_784.astype(np.float32) / 255.0).reshape(784, 1)
    a_out = _forward_prop(x, params)
    return a_out[:, 0]


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
