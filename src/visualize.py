import matplotlib
matplotlib.use("Agg")                 # save to file, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from . import config

sns.set_theme(style="whitegrid")

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=200)
    plt.close(fig)

def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=y, order=[0, 1], ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No diabetes", "Diabetes"])
    ax.set_title("Class Distribution")
    _save(fig, "01_class_distribution.png")

def plot_metric_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    results.drop(columns=[]).plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    ax.legend(loc="lower right")
    _save(fig, "02_metric_comparison.png")

def plot_confusion_matrices(details):
    n = len(details)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for ax, (name, d) in zip(axes[0], details.items()):
        sns.heatmap(d["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    _save(fig, "03_confusion_matrices.png")

def plot_roc_curves(details, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, d in details.items():
        fpr, tpr, _ = roc_curve(y_test, d["y_score"])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend()
    _save(fig, "04_roc_curves.png")

def plot_pr_curves(details, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, d in details.items():
        p, r, _ = precision_recall_curve(y_test, d["y_score"])
        ax.plot(r, p, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend()
    _save(fig, "05_precision_recall_curves.png")

def plot_feature_importance(model, name):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = list(clf.feature_importances_)
    else:
        imp = [abs(value) for value in clf.coef_[0]]
    order = sorted(range(len(imp)), key=lambda index: imp[index])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([config.FEATURES[index] for index in order], [imp[index] for index in order])
    ax.set_title(f"Feature Importance: {name}")
    _save(fig, "06_feature_importance.png")
