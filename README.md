# Heart Disease Prediction App

Academic web application developed as part of a university bachelor thesis on heart-disease prediction with machine-learning models.

The project combines exploratory data analysis, dataset preprocessing, model comparison, and a small predictive demo built with FastAPI and a static frontend. The application is intended to explain the workflow and demonstrate the final binary disease/no-disease model. It is not a medical diagnosis tool.

Live Demo: https://heart-disease-prediction-app-xyae.onrender.com

## Project Contents

- `Bachelor's Thesis.pdf`: full thesis document.
- `EDA .ipynb`: exploratory data analysis and dataset construction.
- `Model Training.ipynb`: model training, validation, and comparison experiments.
- `heart_disease_uci.csv`: original public dataset.
- `modified_dataset_*.csv`: processed datasets generated during EDA.
- `backend/`: FastAPI backend and prediction API.
- `frontend/`: static HTML, CSS, JavaScript, and visual assets.
- `scripts/train_model.py`: script used to train and save the demo model artifact.

## How To Read The Project

Read the thesis PDF if you want the full work explanation. The web app provides a visual summary of the practical work:

1. `Introduction`: project context and thesis abstract.
2. `EDA`: dataset inspection, cleaning, missing-value strategy, and final datasets.
3. `Model training`: validation protocol, classifier comparison, multiclass experiments, and final model choice.
4. `Predictive demo`: interactive binary prediction demo using the saved model artifact.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Web App

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Useful API endpoints:

- `GET /health`: checks whether the model artifact is available.
- `GET /metadata`: returns project/model metadata.
- `POST /predict`: runs the binary prediction demo.

## Train Or Regenerate The Model Artifact

The saved demo model is stored under `backend/artifacts/`. To regenerate it:

```bash
python scripts/train_model.py
```

This trains the binary disease/no-disease demo model using the processed Ill Conditions dataset and saves:

- `backend/artifacts/heart_model.joblib`
- `backend/artifacts/metadata.json`

For the web demo, the data-collection location field is excluded from the artifact because it is not a patient clinical attribute.

## Disclaimer

This project is an academic demonstration based on historical public data. It does not provide medical diagnosis and must not be used for clinical decisions. Always consult a qualified healthcare professional for medical evaluation.
