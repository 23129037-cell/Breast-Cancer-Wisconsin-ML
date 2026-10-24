
from pathlib import Path
import json
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# 1. CAU HINH API
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "svm_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

app = FastAPI(
    title="Breast Cancer AI API",
    description=(
        "API du doan nghien cuu ung thu vu "
        "su dung mo hinh SVM da huan luyen."
    ),
    version="1.0.0",
)

# =========================
# 2. NAP MO HINH
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay mo hinh: {MODEL_PATH}")

if not FEATURE_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay danh sach dac trung: {FEATURE_NAMES_PATH}"
    )

model = joblib.load(MODEL_PATH)

with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
    FEATURE_NAMES = json.load(file)

if len(FEATURE_NAMES) != 30:
    raise ValueError(
        f"Can 30 dac trung, nhung tim thay {len(FEATURE_NAMES)}."
    )


# =========================
# 3. KIEU DU LIEU DAU VAO
# =========================
class PredictionRequest(BaseModel):
    features: dict[str, float]


# =========================
# 4. KIEM TRA API
# =========================
@app.get("/")
def home():
    return {
        "message": "Breast Cancer AI API is running",
        "docs": "/docs",
        "required_features": len(FEATURE_NAMES),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "SVM",
        "feature_count": len(FEATURE_NAMES),
    }


# =========================
# 5. API DU DOAN
# =========================
@app.post("/predict")
def predict(request: PredictionRequest):
    supplied_features = request.features

    missing = [
        name for name in FEATURE_NAMES
        if name not in supplied_features
    ]

    extra = [
        name for name in supplied_features
        if name not in FEATURE_NAMES
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Danh sach dac trung khong hop le.",
                "missing_features": missing,
                "extra_features": extra,
            },
        )

    values = {}

    for name in FEATURE_NAMES:
        value = supplied_features[name]

        if not math.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Dac trung '{name}' phai la so huu han.",
            )

        values[name] = value

    # Dam bao dung thu tu dac trung luc huan luyen
    X_input = pd.DataFrame(
        [[values[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = int(model.predict(X_input)[0])
    probabilities = model.predict_proba(X_input)[0]

    classes = list(model.classes_)
    probability_by_class = {
        str(int(label)): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    label = "Malignant" if prediction == 1 else "Benign"

    return {
        "prediction": prediction,
        "label": label,
        "probability_benign": probability_by_class.get("0"),
        "probability_malignant": probability_by_class.get("1"),
        "model": "SVM",
        "feature_count": len(FEATURE_NAMES),
        "notice": (
            "Ket qua chi phuc vu nghien cuu va hoc tap, "
            "khong thay the chan doan cua bac si."
        ),
    }
