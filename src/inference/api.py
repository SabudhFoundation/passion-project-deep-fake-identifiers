"""
Deepfake Detection API

Run:
    uvicorn src.inference.api:app --reload

Available Endpoints:
    GET  /           - API information
    GET  /health     - Health check
    GET  /methods    - List available detection methods
    POST /predict    - Predict whether an image is real or fake
"""

import traceback
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.inference.predict import (
    DeepfakePredictor,
    ModelNotTrainedError,
    VALID_METHODS,
)

# ============================================================
# Global Predictor Instance
# ============================================================

predictor: DeepfakePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models at startup and release resources at shutdown.
    """
    global predictor

    predictor = DeepfakePredictor()
    print("✅ Deepfake predictor loaded")

    yield

    predictor = None
    print("🛑 Deepfake predictor unloaded")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Deepfake Detection API",
    description=(
        "Detect whether an image is real or fake using classical "
        "machine learning, CNNs, and dual-channel architectures."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# Accuracy Metadata
# ============================================================

METHOD_ACCURACY = {
    "lbp_svm": 65.11,
    "fft_svm": 68.34,
    "glcm_svm": 62.00,
    "combined_svm": 74.87,

    "glcm_mlp": 66.89,
    "lbp_mlp": 79.93,
    "fft_mlp": 62.32,
    "combined_mlp": 82.33,
    "inceptionv3": 87.63,
    "efficientnet": 99.94,
    "dual_cnn": 97.97,
}

# ============================================================
# Response Models
# ============================================================


class HealthResponse(BaseModel):
    status: str


class MethodsResponse(BaseModel):
    methods: list[str]


class PredictionResponse(BaseModel):
    method: str
    label: str
    prediction: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    reported_accuracy: float


# ============================================================
# Utility Routes
# ============================================================


@app.get("/", tags=["Utility"])
def root():
    """
    Root endpoint.
    """
    return {
        "message": "Deepfake Detection API is running",
        "docs": "/docs",
        "health": "/health",
        "methods": "/methods",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Utility"],
)
def health():
    return {"status": "ok"}


@app.get(
    "/methods",
    response_model=MethodsResponse,
    tags=["Utility"],
)
def get_methods():
    return {"methods": VALID_METHODS}


# ============================================================
# Prediction Route
# ============================================================


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
)
async def predict_image(
    file: UploadFile = File(...),
    method: str = Form(...),
):
    """
    Predict whether an uploaded image is real or fake.
    """

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not initialized.",
        )

    if method not in VALID_METHODS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid method '{method}'",
                "valid_methods": VALID_METHODS,
            },
        )

    extension = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False,
    ) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        result = predictor.predict(temp_path, method)

        return PredictionResponse(
            method=method,
            reported_accuracy=METHOD_ACCURACY.get(method, 0.0),
            **result,
        )

    except ModelNotTrainedError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {repr(e)}",
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)