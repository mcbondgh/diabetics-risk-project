from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "brfss_diabetes_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR = BASE_DIR / "reports" / "metrics"

TARGET = "Diabetes_binary"          # 0 = no diabetes, 1 = prediabetes/diabetes

# The 16 predictors used by the model (Chapter 3, Table 3.3): lifestyle, demographic,
# socio-economic, self-reported health status and basic clinical history.
FEATURES = [
    "BMI", "Smoker", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",  "HighBP", "HighChol"
]

# Valid ranges for the model-ready inputs (BRFSS coding), used to validate predictor input.
FEATURE_RANGES = {
    "BMI": (12, 98), "Smoker": (0, 1), "PhysActivity": (0, 1), "Fruits": (0, 1),
    "Veggies": (0, 1), "HvyAlcoholConsump": (0, 1), "GenHlth": (1, 5),
    "MentHlth": (0, 30), "PhysHlth": (0, 30), "DiffWalk": (0, 1), "Sex": (0, 1),
    "Age": (1, 13), "Education": (1, 6), "Income": (1, 8), "HighBP":(0, 1), "HighChol" : (0,1)
}

# Limits for human-entered values before conversion to BRFSS model codes.
RAW_INPUT_RANGES = {
    "weight_kg": (25, 400), "height_cm": (100, 250), "age_years": (18, 110),
}

# Only these features may take non-whole values; every other feature is a coded category or a day count.
CONTINUOUS_FEATURES = {"BMI"}

# ---------------- Training settings (documented in Chapter 3) ----------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3             # Chapter 3 documents stratified 3-fold CV; update the thesis if this changes
USE_SMOTE = True         # False = use class weights instead
SCORING = "f1"           # tuning criterion; "recall" would prioritise catching cases
BEST_MODEL_METRIC = "F1" # metric used to pick the model saved as best_model
DECISION_THRESHOLD = 0.5

# ---------------- Prediction and explanation settings ----------------
# Provisional screening bands, set by hand (not yet derived from data; see Chapter 3, Section 3.10).
RISK_BAND_LOW = 0.30     # score below this = Low
RISK_BAND_HIGH = 0.60    # score at or above this = High; in between = Moderate
DATASET_BASE_RATE = 0.145  # share of positive cases in the training data
MODEL_VERSION = "v1-random-forest-2026.11"
TOP_N_FACTORS = 5        # number of SHAP factors returned per prediction

# Features the user can change; only these generate advice (Chapter 3, Section 3.10.2).
MODIFIABLE_FEATURES = {"Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "BMI"}

# Plain-language names for the interface.
FEATURE_LABELS = {
    "BMI": "Body mass index", "GenHlth": "General health rating",
    "PhysHlth": "Days of poor physical health", "MentHlth": "Days of poor mental health",
    "Age": "Age group", 
    "HighBP": "High blood pressure", "HighChol": "High cholesterol",
    "PhysActivity": "Physical activity", "Smoker": "Smoking history",
    "Fruits": "Daily fruit", "Veggies": "Daily vegetables",
    "HvyAlcoholConsump": "Heavy drinking", "DiffWalk": "Difficulty walking",
    "Sex": "Sex", "Education": "Education level", "Income": "Income band",
}

for d in (MODELS_DIR, FIG_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)
