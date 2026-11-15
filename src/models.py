from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from importlib import import_module
from . import config

try:
    LGBMClassifier = import_module("lightgbm").LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    XGBClassifier = import_module("xgboost").XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    ImbalancedPipeline = import_module("imblearn.pipeline").Pipeline
    SMOTE = import_module("imblearn.over_sampling").SMOTE
except ImportError:
    ImbalancedPipeline = None
    SMOTE = None


def get_model_configs(y_train):
    """Returns {name: (estimator, param_grid)}."""
    cw = None if config.USE_SMOTE else "balanced"
    spw = 1.0 if config.USE_SMOTE else (y_train == 0).sum() / (y_train == 1).sum()
    rs = config.RANDOM_STATE

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight=cw, random_state=rs),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight=cw, random_state=rs, n_jobs=1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [10, 20]},
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                          random_state=rs, n_jobs=1),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1]},
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = (
            LGBMClassifier(class_weight=cw, random_state=rs, n_jobs=1, verbose=-1),
            {"clf__n_estimators": [200, 400], "clf__num_leaves": [31, 63],
             "clf__learning_rate": [0.05, 0.1]},
        )
    return models


def build_pipeline(name, estimator):
    steps = []
    if name == "Logistic Regression":          # linear models need scaled inputs
        steps.append(("scaler", StandardScaler()))
    if config.USE_SMOTE:
        if SMOTE is None:
            raise ImportError("Install 'imbalanced-learn' to use SMOTE.")
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("clf", estimator))
    pipeline = ImbalancedPipeline if config.USE_SMOTE else Pipeline
    if pipeline is None:
        raise ImportError("Install 'imbalanced-learn' to use SMOTE.")
    return pipeline(steps)
