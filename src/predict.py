from importlib import import_module
from numbers import Real
import datetime as dateObject
import pandas as pd

from . import config


class DiabetesPredictor:
    def __init__(self, model_path=None):
        path = model_path or (config.MODELS_DIR / "best_model.joblib")
        try:
            joblib = import_module("joblib")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'joblib' package is required to load the model. "
                "Install it with: python -m pip install joblib"
            ) from exc
        self.model = joblib.load(path)
        # Report the estimator actually loaded from best_model.joblib.
        self.best_model = type(self.model.named_steps["clf"]).__name__
        # Build the SHAP explainer once (slow); it is reused for every prediction.
        self.explainer = self._build_explainer()

    # ---------- helpers so users don't need to know BRFSS codes ----------
    @staticmethod
    def _check_raw(name: str, value) -> None:
        """Reject non-numbers, NaN and values outside the raw-input limits."""
        limits = config.RAW_INPUT_RANGES.get(name, config.FEATURE_RANGES.get(name))
        if limits is None:
            raise KeyError(f"No validation range configured for {name!r}")
        lo, hi = limits
        if isinstance(value, bool) or not isinstance(value, Real) or value != value:
            raise ValueError(f"{name} must be a number, got {value!r}")
        if not lo <= float(value) <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")

    @staticmethod
    def calculate_bmi(weight_kg: float, height_cm: float) -> float:
        DiabetesPredictor._check_raw("weight_kg", weight_kg)
        DiabetesPredictor._check_raw("height_cm", height_cm)
        bmi = weight_kg / (height_cm / 100) ** 2
        DiabetesPredictor._check_raw("BMI", bmi)
        return round(bmi, 1)

    @staticmethod
    def age_to_category(age_years: int) -> int:
        """BRFSS age groups: 1 = 18-24, 2 = 25-29 ... 12 = 75-79, 13 = 80+."""
        DiabetesPredictor._check_raw("age_years", age_years)
        if age_years < 25:
            return 1
        if age_years >= 80:
            return 13
        return int((age_years - 25) // 5 + 2)

    # ---------- validation ----------
    def _validate(self, data: dict) -> None:
        missing = [f for f in config.FEATURES if f not in data]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")
        for f in config.FEATURES:
            v = data[f]
            if isinstance(v, bool):          # JSON true/false from the C# side
                v = int(v)
            if not isinstance(v, Real) or v != v:
                raise ValueError(f"{f} must be a number, got {data[f]!r}")
            lo, hi = config.FEATURE_RANGES[f]
            if not lo <= float(v) <= hi:
                raise ValueError(f"{f} must be between {lo} and {hi}, got {v}")
            if f not in config.CONTINUOUS_FEATURES and float(v) != int(float(v)):
                raise ValueError(f"{f} must be a whole number, got {v}")

    # ---------- explanation ----------
    def _build_explainer(self):
        try:
            shap = import_module("shap")
            return shap.TreeExplainer(self.model.named_steps["clf"])
        except Exception as exc:     # e.g. the selected model is not tree-based
            print(f"SHAP explainer unavailable: {exc}")
            return None

    def _explain(self, row: pd.DataFrame) -> list:
        """Top contributing factors for one person (SHAP values for the diabetes class)."""
        if self.explainer is None:
            return []
        sv = self.explainer.shap_values(row)
        if isinstance(sv, list):             # older SHAP versions: [class0, class1]
            vals = sv[1][0]
        elif sv.ndim == 3:                   # newer SHAP: (rows, features, classes)
            vals = sv[0, :, 1]
        else:
            vals = sv[0]
        top = sorted(zip(config.FEATURES, vals), key=lambda t: abs(t[1]), reverse=True)
        return [
            {
                "feature": f,
                "label": config.FEATURE_LABELS.get(f, f),
                "direction": "increases" if v > 0 else "decreases",
                "modifiable": f in config.MODIFIABLE_FEATURES,
                "contribution": round(float(v), 4),
            }
            for f, v in top[: config.TOP_N_FACTORS]
        ]

    # ---------- main method ----------
    def predict(self, user_input: dict) -> dict:
        self._validate(user_input)
        row = pd.DataFrame([{
            f: int(user_input[f]) if isinstance(user_input[f], bool) else user_input[f]
            for f in config.FEATURES
        }])
        prob = float(self.model.predict_proba(row)[0, 1])
        level = ("Low" if prob < config.RISK_BAND_LOW
                 else "Moderate" if prob < config.RISK_BAND_HIGH else "High")
        wording = {"Low": "Lower estimated risk",
                   "Moderate": "Moderate estimated risk",
                   "High": "Higher estimated risk"}[level]
        return {
            "date:" : str(dateObject.datetime.today().now()),
            "prediction": wording,
            "probability": round(prob, 4),   # kept so existing demo code still works
            "risk_score": round(prob, 4),    # name used in the Chapter 3 API contract
            "risk_level": level,
            "best_model": self.best_model,
            # "dataset_base_rate": config.DATASET_BASE_RATE,
            "factors": self._explain(row),
            "model_version": config.MODEL_VERSION,
            "disclaimer": "Kindly note this tool is for Screening aid only. Not a medical diagnosis.",
        }
