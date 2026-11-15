"""
ML service for the diabetes screening prototype (Chapter 3, Section 3.11.4).

Endpoints
    POST /v1/predict     raw user values in, risk score + band + SHAP factors out
    GET  /v1/model-info  model version, training date, feature set, checksum, thresholds
    GET  /health         liveness check

Run from the project root:
    uvicorn api.app:app --host 127.0.0.1 --port 8000

Environment variables (optional, never hard-code secrets):
    ML_API_KEY       required; callers must send it in the X-API-Key header
    ML_MODEL_SHA256  if set, the service refuses to start unless the model file matches it

Generate a UUID API key with:
    python -c "import uuid; print(uuid.uuid4())"
"""
import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.predict import DiabetesPredictor

log = logging.getLogger("ml-service")
MODEL_FILE = config.MODELS_DIR / "best_model.joblib"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].lstrip()
        key, separator, value = entry.partition("=")
        key = key.strip()
        if not separator or not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ[key] = value


_load_env_file(ENV_FILE)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the frozen model and build the SHAP explainer ONCE, at startup."""
    if not os.getenv("ML_API_KEY", "").strip():
        raise RuntimeError(
            "ML_API_KEY is required. Generate one with: "
            'python -c "import uuid; print(uuid.uuid4())"'
        )
    digest = _sha256(MODEL_FILE)
    expected = os.getenv("ML_MODEL_SHA256")
    if expected and expected.strip().lower() != digest:
        raise RuntimeError("Model checksum mismatch: refusing to start (see ML_MODEL_SHA256).")
    app.state.predictor = DiabetesPredictor(MODEL_FILE)
    app.state.digest = digest
    app.state.trained_on = datetime.fromtimestamp(MODEL_FILE.stat().st_mtime, tz=timezone.utc).date().isoformat()
    yield


app = FastAPI(title="Diabetes screening ML service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- schemas
class AssessmentRequest(BaseModel):
    """Raw user values exactly as listed in Table 3.10. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    ageYears: int = Field(strict=True, ge=18, le=110)
    sex: Literal["female", "male"]
    heightCm: float = Field(strict=True, ge=100, le=250)
    weightKg: float = Field(strict=True, ge=25, le=400)
    smoker: bool = Field(strict=True)
    physicallyActive: bool = Field(strict=True)
    eatsFruitDaily: bool = Field(strict=True)
    eatsVegetablesDaily: bool = Field(strict=True)
    heavyAlcohol: bool = Field(strict=True)
    generalHealth: int = Field(strict=True, ge=1, le=5)
    mentalHealthDays: int = Field(strict=True, ge=0, le=30)
    physicalHealthDays: int = Field(strict=True, ge=0, le=30)
    difficultyWalking: bool = Field(strict=True)
    educationLevel: int = Field(strict=True, ge=1, le=6)
    incomeBand: int = Field(strict=True, ge=1, le=8)
    highBloodPressure: bool = Field(strict=True)
    highCholesterol: bool = Field(strict=True)
    # Consent is handled by the C# gateway; this stateless service accepts and ignores it.
    consentToResearchUse: bool = Field(default=False, strict=True)


class Factor(BaseModel):
    feature: str
    label: str
    direction: Literal["increases", "decreases"]
    modifiable: bool
    contribution: float

class AssessmentResponse(BaseModel):
    bestModel: str
    modelVersion: str
    riskScore: float
    band: Literal["LOW", "MODERATE", "HIGH"]
    # datasetBaseRate: float
    factors: list[Factor]
    disclaimer: str


# ---------------------------------------------------------------- security
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("ML_API_KEY", "")
    print(expected)
    if not expected or not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------- errors
def _errors(items: list[dict]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"errors": items})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    items = []
    for e in exc.errors():
        loc = [str(p) for p in e["loc"] if p != "body"]
        items.append({"field": ".".join(loc) or "body", "message": e["msg"]})
    return _errors(items)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("Unhandled error in %s", request.url.path)   # request bodies are never logged
    return JSONResponse(status_code=500, content={"error": "Internal error in the prediction service"})


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/model-info", dependencies=[Depends(require_api_key)])
def model_info(request: Request):
    s = request.app.state
    return {
        "modelVersion": config.MODEL_VERSION,
        "trainedOn": s.trained_on,                  # model file's last-modified date
        "modelSha256": s.digest,
        "features": config.FEATURES,
        "decisionThreshold": config.DECISION_THRESHOLD,
        "bandCutoffs": {"low": config.RISK_BAND_LOW, "high": config.RISK_BAND_HIGH},
        "datasetBaseRate": config.DATASET_BASE_RATE,
    }


@app.post("/v1/predict", response_model=AssessmentResponse, dependencies=[Depends(require_api_key)])
def predict(body: AssessmentRequest, request: Request):
    p: DiabetesPredictor = request.app.state.predictor
    try:
        bmi = p.calculate_bmi(weight_kg=body.weightKg, height_cm=body.heightCm)
    except ValueError as exc:                        # derived BMI outside 12 to 70
        return _errors([{"field": "heightCm, weightKg", "message": str(exc)}])

    model_input = {
        "BMI": bmi,
        "Smoker": body.smoker,
        "PhysActivity": body.physicallyActive,
        "Fruits": body.eatsFruitDaily,
        "Veggies": body.eatsVegetablesDaily,
        "HvyAlcoholConsump": body.heavyAlcohol,
        "GenHlth": body.generalHealth,
        "MentHlth": body.mentalHealthDays,
        "PhysHlth": body.physicalHealthDays,
        "DiffWalk": body.difficultyWalking,
        "Sex": 1 if body.sex == "male" else 0,      # BRFSS coding: 0 = female, 1 = male
        "Age": p.age_to_category(body.ageYears),
        "Education": body.educationLevel,
        "Income": body.incomeBand,
        "HighBP": body.highBloodPressure,
        "HighChol": body.highCholesterol,
    }
    result = p.predict(model_input)
    return AssessmentResponse(
        bestModel=result["best_model"],
        modelVersion=result["model_version"],
        riskScore=result["risk_score"],
        band=result["risk_level"].upper(),
        # datasetBaseRate=result["dataset_base_rate"],
        factors=result["factors"],
        disclaimer=result["disclaimer"],
    )
