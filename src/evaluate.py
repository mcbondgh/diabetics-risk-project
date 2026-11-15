import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from . import config
import json 


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }
    details = {
        "y_pred": y_pred,
        "y_score": y_score,
        "cm": confusion_matrix(y_test, y_pred),
    }
    return metrics, details


def evaluate_all(models: dict, X_test, y_test):
    rows, details = {}, {}
    for name, model in models.items():
        rows[name], details[name] = evaluate_model(model, X_test, y_test)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, details[name]["y_pred"],
                                    target_names=["No diabetes", "Diabetes"]))
    results = pd.DataFrame(rows).T.round(4)

    # --- CSV ---
    results.to_csv(config.METRICS_DIR / "model_comparison.csv")

    # --- JSON (metrics + confusion matrix per model) ---
    json_output = {}
    for name in results.index:
        cm = details[name]["cm"]
        json_output[name] = {
            **results.loc[name].to_dict(),
            "confusion_matrix": {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1]),
            },
        }
    with open(config.METRICS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)

    return results, details
