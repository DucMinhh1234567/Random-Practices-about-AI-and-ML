import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
 
 
ARTIFACT_PATH = Path("artifacts/breast_cancer_model.joblib")
 
app = FastAPI(
    title="Breast Cancer Prediction API",
    version="1.0.0",
    description="A FastAPI server for serving a scikit-learn breast cancer classifier",
)
 
 
class PredictionRequest(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float
 
 
@app.on_event("startup")
def load_model():
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {ARTIFACT_PATH}. Run `python train.py` first."
        )
 
    artifact = joblib.load(ARTIFACT_PATH)
    app.state.model = artifact["model"]
    app.state.target_names = artifact["target_names"]
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = np.array([[
            request.mean_radius,
            request.mean_texture,
            request.mean_perimeter,
            request.mean_area,
            request.mean_smoothness,
            request.mean_compactness,
            request.mean_concavity,
            request.mean_concave_points,
            request.mean_symmetry,
            request.mean_fractal_dimension,
            request.radius_error,
            request.texture_error,
            request.perimeter_error,
            request.area_error,
            request.smoothness_error,
            request.compactness_error,
            request.concavity_error,
            request.concave_points_error,
            request.symmetry_error,
            request.fractal_dimension_error,
            request.worst_radius,
            request.worst_texture,
            request.worst_perimeter,
            request.worst_area,
            request.worst_smoothness,
            request.worst_compactness,
            request.worst_concavity,
            request.worst_concave_points,
            request.worst_symmetry,
            request.worst_fractal_dimension,
        ]])
 
        model = app.state.model
        target_names = app.state.target_names
 
        prediction_id = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
 
        return {
            "prediction_id": prediction_id,
            "prediction_label": target_names[prediction_id],
            "probabilities": {
                target_names[i]: float(round(probabilities[i], 6))
                for i in range(len(target_names))
            }
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))