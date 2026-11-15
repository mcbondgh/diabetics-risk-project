# Diabetes Risk Assessment

A diabetes risk assessment project using the BRFSS diabetes dataset. It includes
model training and evaluation code, saved model artifacts, reports, and a
FastAPI prediction service. This is a research prototype, not a medical
diagnostic tool.

## Setup

Install the Python dependencies from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The saved models are stored with Git LFS. Install Git LFS before cloning, then
run `git lfs pull` if the model files were not downloaded automatically.

## Run the API

Set an API key in the environment, then launch the service:

```powershell
$env:ML_API_KEY = "<your-api-key>"
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

The raw dataset is in `data/raw/brfss_diabetes_dataset.csv`. Do not use
predictions as a substitute for professional medical advice.
