# DIRISA SDC 2026 — Team VUT: Youth Voter Registration Gap Dashboard

Streamlit scaffold for Challenge 1 (electoral participation ahead of the
2026 Local Government Elections). Built so the UI, filtering, and
prediction flow can be developed in parallel with data collection and
modelling.

## Run it

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## What's real vs. placeholder right now

The app runs out of the box on **synthetic sample data** and a **dummy
heuristic** so there's always something to click through. Two things are
wired up as placeholders for the rest of the team to fill in:

### 1. Dataset → `data/processed/municipality_youth_gap.csv`

Drop the cleaned, merged dataset (output of notebooks
`02_cleaning_merging.ipynb` / `03_feature_engineering.ipynb`) here. Expected
columns:

```
municipality, province, urban_rural, eligible_youth, registered_youth,
youth_registration_gap, gap_growth, registration_momentum,
unemployment_rate, cluster_profile
```

The app detects the file automatically — no code changes needed. The
"Live artifact status" panel on the Methodology tab shows whether it's been
found.

### 2. Model → `models/gap_growth_model.pkl`

Once `05_model.ipynb` produces a trained clustering + regression model,
save it with:

```python
import joblib
joblib.dump(
    {
        "cluster_model": kmeans_model,
        "regression_model": linreg_model,
        "feature_columns": ["eligible_youth", "registered_youth", "urban_rural", "unemployment_rate", "province"],
    },
    "models/gap_growth_model.pkl",
)
```

The Predictions tab will load it automatically and use the real
`.predict()` calls instead of the placeholder heuristic.

## Pages

- **Project Overview** — problem statement, research questions, stakeholders, scope
- **Data Explorer** — search by municipality name, filter by province / urban-rural / cluster profile / gap / unemployment range, sort any column, download filtered CSV, quick bar chart
- **Predictions** — form-based single-municipality prediction, using the real model if present, otherwise a transparent placeholder heuristic
- **Methodology & Data Status** — feature table, known data limitations, live check of whether the real dataset/model files have landed yet