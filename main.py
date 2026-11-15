from src import config
from src.data_loader import load_data, summarize_data, split_data
from src.train import train_all
from src.evaluate import evaluate_all
from src import visualize as viz
import joblib
import pandas as pd

def run_pipeline():
    df = load_data()
    summarize_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    models = train_all(X_train, y_train)
    results, details = evaluate_all(models, X_test, y_test)
    print("\n", results)

    # Save the best model under a fixed name for the predictor
    metric_columns = {str(column).casefold(): column for column in results.columns}
    metric_key = config.BEST_MODEL_METRIC.casefold()
    if metric_key not in metric_columns:
        available = ", ".join(map(str, results.columns))
        raise ValueError(
            f"Unknown best-model metric {config.BEST_MODEL_METRIC!r}. "
            f"Available metrics: {available}"
        )
    metric_column = metric_columns[metric_key]
    best_name = results[metric_column].idxmax()
    joblib.dump(models[best_name], config.MODELS_DIR / "best_model.joblib")
    print(f"\nBest model by {metric_column}: {best_name}")

    # Graphs
    viz.plot_class_distribution(df[config.TARGET])
    viz.plot_metric_comparison(results)
    viz.plot_confusion_matrices(details)
    viz.plot_roc_curves(details, y_test)
    viz.plot_pr_curves(details, y_test)
    viz.plot_feature_importance(models[best_name], best_name)


def demo_prediction():
    from src.predict import DiabetesPredictor
    p = DiabetesPredictor()
    sample = {
        "BMI": p.calculate_bmi(weight_kg=200, height_cm=190),
        "Smoker": 1, "PhysActivity":0, "Fruits": 1, 
        "Veggies": 1,"HvyAlcoholConsump": 0, 
        "GenHlth": 5, "MentHlth": 1, "PhysHlth": 20,
        "DiffWalk": 1, "Sex": 0, "Age": p.age_to_category(18),
        "Education": 4, "Income": 5, "HighBP": 0, "HighChol": 0
   }
    print(p.predict(sample))


def quickTest():
    model = joblib.load("models/best_model.joblib")

    # The steps in the pipeline
    print(model.named_steps)

    # The tuned hyperparameters the grid search chose
    print(model.named_steps["clf"].get_params())

    sample = pd.DataFrame([{
        "BMI": 31, "Smoker": 1, "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
        "HvyAlcoholConsump": 0, "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10,
        "DiffWalk": 1, "Sex": 1, "Age": 9, "Education": 4, "Income": 3,
        "HighBP" : 1, "HighChol" : 0
    }])
    print(model.predict_proba(sample))


def main() -> None:
    # run_pipeline()
    demo_prediction()
    # quickTest()

if __name__ == "__main__":
    main()
