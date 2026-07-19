# Food Delivery Time Predictor

A simple Streamlit app that predicts food delivery time using a trained Random Forest regression model.

## Files

- `app.py` — Streamlit application.
- `delivery time prediction.ipynb` — exploratory notebook for data analysis and model training.
- `rf_model (1).pkl` — serialized Random Forest model used by the app.
- `Food_Delivery_Times.csv.xlsx` — dataset source file.
- `requirements.txt` — Python dependencies.
- `.gitignore` — files and folders to ignore in Git.

## Setup

1. Create and activate a Python virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- The app loads `rf_model (1).pkl` at runtime, so this file must remain in the project root.
- The notebook includes data exploration and model development code.
