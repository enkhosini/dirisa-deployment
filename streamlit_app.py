"""
DIRISA SDC 2026 — Youth Voter Registration Gap Dashboard
Team VUT | Challenge 1: Electoral participation and representation

This app is a scaffold for the team's submission. Two pieces are still being
built by the rest of the team and are wired in here as clearly-marked
placeholders:

  1. DATASET  -> data/processed/municipality_youth_gap.csv
     (the cleaned, merged IEC + Census + QLFS dataset described in the
     technical document's Section 5/7.2/7.3)

  2. MODEL    -> models/gap_growth_model.pkl
     (the trained clustering + regression artifact described in Section 7.5,
     saved with joblib as a dict: {"cluster_model": ..., "regression_model": ...,
     "feature_columns": [...]})

Until those files exist, the app runs on a synthetic sample so the rest of
the team can build the UI, filtering, and prediction flow in parallel with
data collection and modelling. Every place that uses fake data says so.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Paths (placeholders — point these at the real artifacts once they exist)
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "municipality_youth_gap.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gap_growth_model.pkl"

st.set_page_config(
    page_title="DIRISA SDC 2026 — Youth Registration Gap",
    page_icon="🗳️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "municipality",
    "province",
    "urban_rural",
    "eligible_youth",
    "registered_youth",
    "youth_registration_gap",
    "gap_growth",
    "registration_momentum",
    "unemployment_rate",
    "cluster_profile",
]


@st.cache_data
def load_sample_dataset(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    SYNTHETIC PLACEHOLDER DATA.
    Structurally matches the feature table in Section 7.3 of the technical
    document, but the numbers and municipality names are randomly generated
    — not real IEC / Census / QLFS figures. Replace by dropping the real
    cleaned dataset at DATASET_PATH.
    """
    rng = np.random.default_rng(seed)
    provinces = [
        "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
        "Free State", "Limpopo", "Mpumalanga", "North West", "Northern Cape",
    ]
    rows = []
    for i in range(n):
        province = rng.choice(provinces)
        urban_rural = rng.choice(["Urban", "Rural", "Mixed"], p=[0.4, 0.35, 0.25])
        eligible_youth = int(rng.integers(8_000, 220_000))
        registration_rate = rng.uniform(0.35, 0.92)
        registered_youth = int(eligible_youth * registration_rate)
        gap = (eligible_youth - registered_youth) / eligible_youth
        gap_growth = round(rng.normal(0.03, 0.05), 4)
        momentum = round(rng.normal(0.02, 0.08), 4)
        unemployment_rate = round(rng.uniform(0.18, 0.55), 3)
        if gap > 0.5 and gap_growth > 0.03:
            profile = "high gap / widening"
        elif gap > 0.5:
            profile = "high gap / stable"
        elif gap_growth < 0:
            profile = "low gap / closing"
        else:
            profile = "moderate gap / stable"
        rows.append(
            {
                "municipality": f"Sample Municipality {i + 1:02d}",
                "province": province,
                "urban_rural": urban_rural,
                "eligible_youth": eligible_youth,
                "registered_youth": registered_youth,
                "youth_registration_gap": round(gap, 4),
                "gap_growth": gap_growth,
                "registration_momentum": momentum,
                "unemployment_rate": unemployment_rate,
                "cluster_profile": profile,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_dataset():
    """
    Returns (dataframe, is_placeholder).
    Tries the real processed dataset first; falls back to the synthetic
    sample so the app always has something to display.
    """
    if DATASET_PATH.exists():
        try:
            df = pd.read_csv(DATASET_PATH)
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                st.warning(
                    f"Loaded {DATASET_PATH.name}, but it's missing expected "
                    f"columns: {', '.join(missing)}. Showing it as-is."
                )
            return df, False
        except Exception as e:
            st.error(f"Could not read {DATASET_PATH}: {e}. Falling back to sample data.")
            return load_sample_dataset(), True
    return load_sample_dataset(), True


def load_model():
    """
    PLACEHOLDER MODEL LOADER.
    Looks for a joblib-pickled dict at MODEL_PATH with keys:
      "cluster_model", "regression_model", "feature_columns"
    Returns None if the trained model doesn't exist yet.
    """
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Found {MODEL_PATH.name} but couldn't load it: {e}")
        return None


def dummy_predict(row: dict) -> dict:
    """
    PLACEHOLDER PREDICTION LOGIC — stands in for the real regression +
    clustering model (Section 7.5). This is a simple, transparent heuristic,
    not a trained model. Replace this function's body once MODEL_PATH exists
    by calling the real model's .predict() instead.
    """
    base_gap_growth = (
        0.10
        + 0.06 * row["unemployment_rate"]
        - 0.04 * (row["urban_rural"] == "Urban")
        + 0.05 * (row["urban_rural"] == "Rural")
    )
    predicted_gap_growth = round(base_gap_growth, 4)
    if predicted_gap_growth > 0.08:
        predicted_profile = "high gap / widening"
    elif predicted_gap_growth > 0.03:
        predicted_profile = "moderate gap / stable"
    else:
        predicted_profile = "low gap / closing"
    return {
        "predicted_gap_growth": predicted_gap_growth,
        "predicted_cluster_profile": predicted_profile,
    }


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.title("🗳️ Youth Voter Registration Gap — Team VUT")
st.caption("DIRISA SDC 2026 · Challenge 1 · Electoral participation ahead of the 2026 Local Government Elections")

tab_overview, tab_explorer, tab_predict, tab_methodology = st.tabs(
    ["📖 Project Overview", "🔎 Data Explorer", "🔮 Predictions", "🧪 Methodology & Data Status"]
)

# ---- Tab 1: Project Overview ---------------------------------------------
with tab_overview:
    st.header("What this project is")
    st.markdown(
        """
South African municipal electoral planning currently lacks an integrated,
predictive framework for identifying **localized youth voter disengagement**.
Registration reporting today is a static snapshot — it isn't cross-referenced
against historical turnout trends or population baselines, which makes it
hard for election bodies and civic groups to proactively target outreach and
registration drives.

**Research questions**
- *Primary:* Which South African municipalities exhibit the widest and
  fastest-growing gap between eligible youth citizens and registered youth
  voters?
- *Secondary:* How will youth voter turnout impact overall municipal
  participation in the 2026 Local Government Elections?
        """
    )

    st.subheader("Why it matters")
    stakeholders = pd.DataFrame(
        [
            ("IEC", "Prioritise registration drives and civic education where the youth gap is largest and widening fastest"),
            ("Civil society / NGOs", "Target outreach where disengagement is structurally worst"),
            ("Political parties", "Understand where youth participation is collapsing"),
            ("Journalists", "Evidence base for reporting beyond national averages"),
            ("Policymakers", "Connect disengagement to socioeconomic conditions"),
        ],
        columns=["Stakeholder", "How they benefit"],
    )
    st.table(stakeholders)

    st.subheader("Scope")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**In scope**
- All 257 SA local, district & metro municipalities
- Youth = ages 18–34 (IEC bands), cross-referenced with Census 2022's 15–34 band
- 2021 registration baseline vs. 2026 pre-election snapshot
- Municipality-level analysis
            """
        )
    with col2:
        st.markdown(
            """
**Out of scope**
- Individual-level voter behaviour prediction
- Party-political analysis or vote-share forecasting
- Real-time election-night prediction
            """
        )

    st.subheader("Evidence of urgency")
    st.markdown(
        """
- Youth (20–29) turnout of registered voters fell from **50% in 2016** to **35% in 2021**
- Only **15%** of 18–19 year olds were registered in 2021, down from 30% in 2005
- **81%** of SA youth are dissatisfied with how democracy works; **80%** distrust national government
- 2026 registration surges show uneven geographic momentum — some municipalities surge, others stagnate
        """
    )

    st.subheader("Deployment plan")
    st.markdown(
        """
This Streamlit app is the planned deployment surface: an interactive
choropleth map, a sortable/filterable ranking table, municipality drill-down,
a top-20 priority list, and this methodology tab — backed by a GitHub repo
with reproduction instructions.
        """
    )

# ---- Tab 2: Data Explorer --------------------------------------------------
with tab_explorer:
    df, is_placeholder = load_dataset()

    if is_placeholder:
        st.info(
            f"⚠️ **Showing synthetic sample data**, not the real IEC/Census/QLFS "
            f"dataset. Drop the cleaned dataset at `{DATASET_PATH.relative_to(PROJECT_ROOT)}` "
            f"(columns: {', '.join(EXPECTED_COLUMNS)}) to replace it automatically.",
            icon="⚠️",
        )

    st.header("Search, filter & sort")

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            search_term = st.text_input("Search municipality name")
            provinces_selected = st.multiselect(
                "Province", sorted(df["province"].dropna().unique()) if "province" in df else []
            )
        with f2:
            urban_rural_selected = st.multiselect(
                "Urban / Rural", sorted(df["urban_rural"].dropna().unique()) if "urban_rural" in df else []
            )
            profiles_selected = st.multiselect(
                "Cluster profile", sorted(df["cluster_profile"].dropna().unique()) if "cluster_profile" in df else []
            )
        with f3:
            if "youth_registration_gap" in df:
                gap_min, gap_max = float(df["youth_registration_gap"].min()), float(df["youth_registration_gap"].max())
                gap_range = st.slider("Youth registration gap", gap_min, gap_max, (gap_min, gap_max))
            else:
                gap_range = None
            if "unemployment_rate" in df:
                un_min, un_max = float(df["unemployment_rate"].min()), float(df["unemployment_rate"].max())
                un_range = st.slider("Unemployment rate", un_min, un_max, (un_min, un_max))
            else:
                un_range = None

    filtered = df.copy()
    if search_term:
        filtered = filtered[filtered["municipality"].str.contains(search_term, case=False, na=False)]
    if provinces_selected:
        filtered = filtered[filtered["province"].isin(provinces_selected)]
    if urban_rural_selected:
        filtered = filtered[filtered["urban_rural"].isin(urban_rural_selected)]
    if profiles_selected:
        filtered = filtered[filtered["cluster_profile"].isin(profiles_selected)]
    if gap_range:
        filtered = filtered[filtered["youth_registration_gap"].between(*gap_range)]
    if un_range:
        filtered = filtered[filtered["unemployment_rate"].between(*un_range)]

    sort_col1, sort_col2 = st.columns([2, 1])
    with sort_col1:
        sort_column = st.selectbox("Sort by", options=list(filtered.columns), index=list(filtered.columns).index("youth_registration_gap") if "youth_registration_gap" in filtered.columns else 0)
    with sort_col2:
        sort_desc = st.toggle("Descending", value=True)

    filtered = filtered.sort_values(by=sort_column, ascending=not sort_desc)

    st.caption(f"Showing {len(filtered)} of {len(df)} municipalities")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️ Download filtered results as CSV",
        data=csv_buffer.getvalue(),
        file_name="filtered_municipalities.csv",
        mime="text/csv",
    )

    if len(filtered):
        st.subheader("Quick chart")
        chart_metric = st.selectbox(
            "Metric to chart by municipality",
            [c for c in ["youth_registration_gap", "gap_growth", "registration_momentum", "unemployment_rate"] if c in filtered.columns],
        )
        chart_df = filtered.set_index("municipality")[[chart_metric]].head(30)
        st.bar_chart(chart_df)

# ---- Tab 3: Predictions ----------------------------------------------------
with tab_predict:
    st.header("Model predictions")
    model = load_model()

    if model is None:
        st.warning(
            f"⚠️ **No trained model found yet.** This tab is running on a placeholder "
            f"heuristic, not the real clustering + regression model described in the "
            f"technical document (Section 7.5). Once trained, save it with `joblib.dump(...)` "
            f"to `{MODEL_PATH.relative_to(PROJECT_ROOT)}` as a dict with keys "
            f"`cluster_model`, `regression_model`, and `feature_columns`, and this tab "
            f"will use it automatically.",
            icon="⚠️",
        )
    else:
        st.success("✅ Trained model loaded from disk.")

    st.subheader("Try a prediction")
    st.caption("Enter municipality-level features to get a predicted gap-growth and disengagement profile.")

    p1, p2, p3 = st.columns(3)
    with p1:
        in_eligible = st.number_input("Eligible youth (15–34)", min_value=0, value=50_000, step=1000)
        in_registered = st.number_input("Registered youth (18–34)", min_value=0, value=35_000, step=1000)
    with p2:
        in_urban_rural = st.selectbox("Urban / Rural", ["Urban", "Rural", "Mixed"])
        in_unemployment = st.slider("Unemployment rate", 0.0, 0.8, 0.30)
    with p3:
        in_province = st.selectbox(
            "Province",
            ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Free State",
             "Limpopo", "Mpumalanga", "North West", "Northern Cape"],
        )

    if st.button("Predict", type="primary"):
        input_row = {
            "eligible_youth": in_eligible,
            "registered_youth": in_registered,
            "urban_rural": in_urban_rural,
            "unemployment_rate": in_unemployment,
            "province": in_province,
        }
        if model is not None:
            try:
                feature_cols = model.get("feature_columns", list(input_row.keys()))
                X = pd.DataFrame([input_row])[feature_cols]
                reg = model["regression_model"]
                clust = model["cluster_model"]
                predicted_gap_growth = float(reg.predict(X)[0])
                predicted_cluster = clust.predict(X)[0]
                result = {
                    "predicted_gap_growth": round(predicted_gap_growth, 4),
                    "predicted_cluster_profile": predicted_cluster,
                }
            except Exception as e:
                st.error(f"Real model failed to predict ({e}). Falling back to placeholder heuristic.")
                result = dummy_predict(input_row)
        else:
            result = dummy_predict(input_row)

        r1, r2 = st.columns(2)
        r1.metric("Predicted gap growth", f"{result['predicted_gap_growth']:.1%}")
        r2.metric("Predicted disengagement profile", result["predicted_cluster_profile"])

        current_gap = (in_eligible - in_registered) / in_eligible if in_eligible else 0
        st.caption(f"Current youth registration gap for this input: {current_gap:.1%}")

# ---- Tab 4: Methodology & Data Status --------------------------------------
with tab_methodology:
    st.header("Methodology summary")
    st.markdown(
        """
**Feature engineering**

| Feature | Formula | Purpose |
|---|---|---|
| `eligible_youth` | Census 2022 population aged 15–34 | Denominator |
| `registered_youth` | IEC dashboard, ages 18–34 | Numerator |
| `youth_registration_gap` | (eligible − registered) / eligible | Core metric |
| `gap_growth` | gap_2026 − gap_2021 | "Fastest-growing" dimension |
| `registration_momentum` | (reg_2026 − reg_2021) / reg_2021 | Captures surge effect |
| `urban_rural` | Census classification | Contextual feature |
| `unemployment_rate` | Stats SA QLFS | Socioeconomic feature |
| `province` | Categorical | Regional control |

**Model approach:** unsupervised K-means clustering (municipality profiling) +
multiple linear regression (explaining gap growth) + a composite risk score
for priority ranking. See the technical document, Section 7, for the full
justification and evaluation plan (silhouette score, bootstrap resampling,
R², VIF, sensitivity analysis).
        """
    )

    st.header("Known data limitations")
    st.markdown(
        """
- IEC registration dashboard isn't directly downloadable — extraction risk mitigated via scripted scraping + cross-validation against SANEF
- Census 2022 age bands (15–34) differ from IEC bands — using 15–34 as the functional youth definition, with noted imprecision for the 18–19 band
- A 2026 registration surge may not persist to election day — the gap is framed as a structural indicator, not a turnout forecast
- Municipal boundary changes may affect cross-year comparisons — affected municipalities will be flagged or excluded
- Ward-level data is scarce — analysis is scoped to municipality level; ward-level is future work
        """
    )

    st.header("Live artifact status")
    ds_status = "✅ Found" if DATASET_PATH.exists() else "⬜ Not yet added"
    model_status = "✅ Found" if MODEL_PATH.exists() else "⬜ Not yet added"
    status_df = pd.DataFrame(
        [
            ("Dataset", str(DATASET_PATH.relative_to(PROJECT_ROOT)), ds_status),
            ("Trained model", str(MODEL_PATH.relative_to(PROJECT_ROOT)), model_status),
        ],
        columns=["Artifact", "Expected path", "Status"],
    )
    st.table(status_df)
    st.caption(
        "This tab reads the filesystem directly, so it will flip to ✅ automatically "
        "once teammates drop the real files in place — no code changes needed."
    )