# explain_shap.py - run AFTER main.py has trained and saved models/best_model.joblib
from pathlib import Path
from importlib import import_module
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # write image files only, no pop-up windows
import matplotlib.pyplot as plt
shap = import_module("shap")

from src import config
from src.data_loader import load_data, split_data

FIG_DIR = Path("reports/figures")
MET_DIR = Path("reports/metrics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# 1. Same data, same split as training (so these are genuine unseen test rows)
df = load_data()
_, X_test, _, y_test = split_data(df)
feature_names = list(X_test.columns)
X_sample = X_test.sample(2000, random_state=42)

# 2. Load the saved pipeline and pull the Random Forest out of it
model = joblib.load(config.MODELS_DIR / "best_model.joblib")
rf = model.named_steps["clf"]
print("Explaining model:", type(rf).__name__)

# 3. SHAP values for the diabetes class (class 1)
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(X_sample)
if isinstance(sv, list):                   # older SHAP: [class0, class1]
    sv_pos, base = sv[1], explainer.expected_value[1]
elif sv.ndim == 3:                         # newer SHAP: (rows, features, classes)
    sv_pos, base = sv[:, :, 1], explainer.expected_value[1]
else:
    sv_pos, base = sv, explainer.expected_value

# 4. Figures
shap.summary_plot(sv_pos, X_sample, feature_names=feature_names, show=False)
plt.savefig(FIG_DIR / "07_shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()

shap.summary_plot(sv_pos, X_sample, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig(FIG_DIR / "08_shap_bar.png", dpi=200, bbox_inches="tight"); plt.close()

shap.plots.waterfall(
    shap.Explanation(values=sv_pos[0], base_values=base,
                     data=X_sample.iloc[0].values, feature_names=feature_names),
    show=False)
plt.savefig(FIG_DIR / "09_shap_waterfall_example.png", dpi=200, bbox_inches="tight"); plt.close()

# 5. Table of numbers for the thesis text
(pd.DataFrame({"feature": feature_names,
               "mean_abs_shap": np.abs(sv_pos).mean(axis=0)})
   .sort_values("mean_abs_shap", ascending=False)
   .to_csv(MET_DIR / "shap_feature_importance.csv", index=False))

print("Done. Figures in", FIG_DIR, "| table in", MET_DIR / "shap_feature_importance.csv")