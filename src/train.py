import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from . import config
from .models import get_model_configs, build_pipeline


def train_model(name, estimator, grid, X_train, y_train):
    pipe = build_pipeline(name, estimator)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    search = GridSearchCV(pipe, grid, scoring=config.SCORING, cv=cv,
                          n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    print(f"[{name}] best CV {config.SCORING}: {search.best_score_:.4f} "
          f"| params: {search.best_params_}")
    return search.best_estimator_


def train_all(X_train, y_train) -> dict:
    trained = {}
    for name, (estimator, grid) in get_model_configs(y_train).items():
        model = train_model(name, estimator, grid, X_train, y_train)
        joblib.dump(model, config.MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib")
        trained[name] = model
    return trained