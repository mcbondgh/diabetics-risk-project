import importlib

import joblib
import pandas as pd

shap = importlib.import_module("shap")

def explain_tree(X_test):
    """Compute class-1 SHAP values for a caller-provided test DataFrame."""
    best_model = joblib.load("models/best_model.joblib")
    rf = best_model.named_steps["clf"]

    feature_names = X_test.columns.tolist()
    X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_sample)

    # SHAP versions differ in output shape; select class 1 (diabetes).
    if isinstance(sv, list):
        sv_pos = sv[1]
        base = explainer.expected_value[1]
    elif sv.ndim == 3:
        sv_pos = sv[:, :, 1]
        base = explainer.expected_value[1]
    else:
        sv_pos = sv
        base = explainer.expected_value

    return feature_names, X_sample, sv_pos, base