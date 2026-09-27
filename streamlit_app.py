"""
DIRISA SDC 2026 — Youth Voter Registration Gap Dashboard
Team VUT | Challenge 1: Electoral participation and representation

Loads the REAL trained artifacts from the Hugging Face Hub:

  MODEL_REPO   -> smmdlovu/dirisa-sdc-2026-youth-gap-model
    cluster_scaler.joblib + cluster_kmeans.joblib   (Model A — clustering)
    turnout_ridge_model.joblib                       (Model B — regression)

  DATASET_REPO -> smmdlovu/dirisa-sdc-2026-youth-gap-data
    municipality_master_dataset.csv
    (assumed name, matching MODEL_REPO's naming convention — update below
    if your dataset repo is named differently)

IMPORTANT: these are scikit-learn artifacts (joblib), not a transformers
text-generation model. There's no tokenizer or transformer config in this
repo, so `AutoTokenizer` / `AutoModelForCausalLM` will not work against it —
this loads them the correct way, with `huggingface_hub.hf_hub_download` +
`joblib.load`.

Falls back to local bundled files (same directory as this script) if the
Hub is unreachable, so the app still runs offline / while iterating.
"""

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Repo IDs — your real pushed model repo, plus the dataset repo (update the
# dataset repo name if yours differs)
# --------------------------------------------------------------------------
MODEL_REPO = "smmdlovu/dirisa-sdc-2026-youth-gap-model"
DATASET_REPO = "smmdlovu/dirisa-sdc-2026-youth-gap-data"
DATASET_FILE = "municipality_master_dataset.csv"

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="DIRISA SDC 2026 — Youth Registration Gap",
    page_icon="🗳️",
    layout="wide",
)

CLUSTER_COLORS = {
    "High gap / Widening": "#c0392b",
    "High gap / Stable": "#e67e22",
    "Low gap / Widening": "#f1c40f",
    "Low gap / Stable": "#27ae60",
    "Insufficient data": "#95a5a6",
}

# --------------------------------------------------------------------------
# Data / model loading — Hub first, local fallback
# --------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_label)."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
        return pd.read_csv(path), f"Hugging Face Hub ({DATASET_REPO})"
    except Exception:
        local_path = APP_DIR / DATASET_FILE
        if local_path.exists():
            return pd.read_csv(local_path), "local bundled file (Hub unreachable)"
        raise FileNotFoundError(
            f"Could not load dataset from Hub ({DATASET_REPO}) or locally ({local_path})."
        )


@st.cache_resource
def load_models():
    """Returns (cluster_scaler, cluster_kmeans, cluster_config,
    turnout_ridge, turnout_config, source_label) or Nones + error string."""
    filenames = [
        "cluster_scaler.joblib", "cluster_kmeans.joblib", "cluster_model_config.json",
        "turnout_ridge_model.joblib", "turnout_model_config.json",
    ]
    try:
        from huggingface_hub import hf_hub_download
        paths = {f: hf_hub_download(MODEL_REPO, f) for f in filenames}
        source = f"Hugging Face Hub ({MODEL_REPO})"
    except Exception:
        paths = {f: APP_DIR / f for f in filenames}
        if not all(Path(p).exists() for p in paths.values()):
            return (None,) * 5 + (None,)
        source = "local bundled files (Hub unreachable)"

    cluster_scaler = joblib.load(paths["cluster_scaler.joblib"])
    cluster_kmeans = joblib.load(paths["cluster_kmeans.joblib"])
    cluster_config = json.load(open(paths["cluster_model_config.json"]))
    turnout_ridge = joblib.load(paths["turnout_ridge_model.joblib"])
    turnout_config = json.load(open(paths["turnout_model_config.json"]))
    return cluster_scaler, cluster_kmeans, cluster_config, turnout_ridge, turnout_config, source


df, data_source = load_dataset()
cluster_scaler, cluster_kmeans, cluster_config, turnout_ridge, turnout_config, model_source = load_models()
models_loaded = cluster_scaler is not None

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.title("🗳️ Youth Voter Registration Gap — Team VUT")
st.caption("DIRISA SDC 2026 · Challenge 1 · Electoral participation ahead of the 2026 Local Government Elections")

tab_overview, tab_explorer, tab_cluster, tab_turnout, tab_methodology = st.tabs(
    ["📖 Project Overview", "🔎 Data Explorer", "🧭 Cluster Assignment (Q1)",
     "📈 Turnout Impact (Q2)", "🧪 Methodology & Data Status"]
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
**In scope (this pilot)**
- Eastern Cape — 33 municipalities
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
- National rollout (257 municipalities) — future work
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

# ---- Tab 2: Data Explorer --------------------------------------------------
with tab_explorer:
    st.header("Search, filter & sort")
    st.caption(f"Real Eastern Cape pilot dataset — {len(df)} municipalities. Source: {data_source}")

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            search_term = st.text_input("Search municipality name")
            district_selected = st.multiselect(
                "District", sorted(df["district"].dropna().unique()) if "district" in df else []
            )
        with f2:
            profiles_selected = st.multiselect(
                "Cluster profile", sorted(df["cluster_name"].dropna().unique()) if "cluster_name" in df else []
            )
        with f3:
            if "youth_gap_2026" in df:
                gap_min, gap_max = float(df["youth_gap_2026"].min()), float(df["youth_gap_2026"].max())
                gap_range = st.slider("Youth registration gap", gap_min, gap_max, (gap_min, gap_max))
            else:
                gap_range = None

    filtered = df.copy()
    if search_term:
        filtered = filtered[filtered["municipality"].str.contains(search_term, case=False, na=False)]
    if district_selected:
        filtered = filtered[filtered["district"].isin(district_selected)]
    if profiles_selected:
        filtered = filtered[filtered["cluster_name"].isin(profiles_selected)]
    if gap_range:
        filtered = filtered[filtered["youth_gap_2026"].between(*gap_range)]

    sort_col1, sort_col2 = st.columns([2, 1])
    with sort_col1:
        default_sort = "priority_rank" if "priority_rank" in filtered.columns else filtered.columns[0]
        sort_column = st.selectbox("Sort by", options=list(filtered.columns), index=list(filtered.columns).index(default_sort))
    with sort_col2:
        sort_desc = st.toggle("Descending", value=False)

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
            [c for c in ["youth_gap_2026", "youth_registration_change_pct", "youth_disengagement_risk"] if c in filtered.columns],
        )
        chart_df = filtered.set_index("municipality")[[chart_metric]].head(30)
        st.bar_chart(chart_df)

# ---- Tab 3: Cluster Assignment (Model A, Q1) -------------------------------
with tab_cluster:
    st.header("Assign a municipality profile")
    st.caption("Which municipalities have the widest and fastest-growing youth registration gap?")

    if not models_loaded:
        st.warning(
            f"⚠️ **Model artifacts not found** on the Hub ({MODEL_REPO}) or locally. "
            f"Check the repo ID / your internet connection.",
            icon="⚠️",
        )
    else:
        st.success(f"✅ Real KMeans clustering model loaded (silhouette = {cluster_config['silhouette_score']:.3f}). Source: {model_source}")

        feature_cols = cluster_config["feature_cols"]
        inputs = {}
        cols = st.columns(len(feature_cols))
        for col, feat in zip(cols, feature_cols):
            default_val = float(df[feat].median()) if feat in df.columns else 0.0
            inputs[feat] = col.number_input(feat, value=default_val, format="%.4f")

        if st.button("Assign cluster", type="primary"):
            row_df = pd.DataFrame([inputs])[feature_cols]
            X_scaled = cluster_scaler.transform(row_df)
            cluster_id = int(cluster_kmeans.predict(X_scaled)[0])
            cluster_name = cluster_config["cluster_id_to_name"].get(str(cluster_id), "Unknown")
            color = CLUSTER_COLORS.get(cluster_name, "#95a5a6")
            st.markdown(
                f"""<div style="background:{color}; color:white; padding:12px 16px;
                border-radius:6px; display:inline-block; font-weight:bold; margin-top:10px;">
                Assigned cluster: {cluster_name}</div>""",
                unsafe_allow_html=True,
            )

    st.subheader("Top priority municipalities (from the dataset's own ranking)")
    if "priority_rank" in df.columns:
        top = df.sort_values("priority_rank").head(10)
        st.dataframe(
            top[["municipality", "district", "cluster_name", "youth_gap_2026", "priority_rank"]],
            use_container_width=True, hide_index=True,
        )

# ---- Tab 4: Turnout Impact (Model B, Q2) -----------------------------------
with tab_turnout:
    st.header("Estimate turnout impact")
    st.caption("How will youth voter turnout impact overall municipal participation?")

    if not models_loaded:
        st.warning(f"⚠️ **Model artifacts not found** on the Hub ({MODEL_REPO}) or locally.", icon="⚠️")
    else:
        r2 = turnout_config["loocv_r2"]
        mae = turnout_config["loocv_mae_pp"]
        st.warning(
            f"⚠️ **Read this model's output as a rough directional signal, not a forecast.** "
            f"Leave-One-Out CV R² = **{r2:.3f}** (essentially no reliable predictive power), "
            f"MAE = {mae:.2f} percentage points. See the Methodology tab for why.",
            icon="⚠️",
        )

        feature_cols = turnout_config["feature_cols"]
        inputs = {}
        cols = st.columns(len(feature_cols))
        for col, feat in zip(cols, feature_cols):
            default_val = float(df[feat].median()) if feat in df.columns else 0.0
            inputs[feat] = col.number_input(feat, value=default_val, format="%.4f", key=f"turnout_{feat}")

        if st.button("Estimate turnout change", type="primary"):
            row_df = pd.DataFrame([inputs])[feature_cols]
            estimate = float(turnout_ridge.predict(row_df.values)[0])
            st.metric("Estimated overall turnout change", f"{estimate:+.2f} pp")
            st.caption(
                "This is the fitted Ridge regression's output for the scenario above — "
                "an elasticity estimate given the historical 2016→2021 relationship, "
                "not a prediction of actual 2026 turnout."
            )

# ---- Tab 5: Methodology & Data Status --------------------------------------
with tab_methodology:
    st.header("Methodology summary")
    st.markdown(
        """
**Model A — clustering (answers Q1)**

| Feature | Meaning |
|---|---|
| `youth_gap_2026` | (eligible_youth − registered_youth) / eligible_youth |
| `youth_registration_rate_2026` | registered_youth / eligible_youth |
| `youth_registration_change_pct` | District-level youth registration trend, 2016→2021 |
| `turnout_change_2016_2021_pp` | Turnout change, percentage points, 2016→2021 |
| `registered_growth_2016_2021_pct` | Registered voter growth, 2016→2021 |

`StandardScaler` + `KMeans(k=4)`, unsupervised. Silhouette = 0.371.

**Model B — regression (answers Q2)**

Ridge regression: `turnout_change_2016_2021_pp` from `youth_registration_change_pct`,
`registered_growth_2016_2021_pct`, `youth_share_pct_2022`. n=24, evaluated with
Leave-One-Out CV (the only honest evaluation at this sample size).
**LOOCV R² = 0.027** — this explains almost none of the out-of-sample variance.
The one directionally sensible signal is the coefficient on
`youth_registration_change_pct` (positive) — worth investigating further with
more data, not a number to forecast from.
        """
    )

    st.header("Known data limitations")
    st.markdown(
        """
- IEC registration dashboard isn't directly downloadable — extraction risk mitigated via scripted scraping + cross-validation against SANEF
- Census 2022 age bands (15–34) differ from IEC bands — using the functional youth definition, with noted imprecision for the 18–19 band
- A 2026 registration surge may not persist to election day — the gap is framed as a structural indicator, not a turnout forecast
- 9 of 33 municipalities lack district-level trend data and are labelled "Insufficient data" rather than force-clustered
- Ward-level data is scarce — analysis is scoped to municipality level
- Analysis is currently Eastern Cape only (pilot province)
        """
    )

    st.header("Live artifact status")
    status_df = pd.DataFrame(
        [
            ("Dataset", DATASET_REPO, data_source if 'data_source' in dir() else "not loaded"),
            ("Models (cluster + turnout)", MODEL_REPO, model_source if models_loaded else "⚠️ not found"),
        ],
        columns=["Artifact", "Repo", "Loaded from"],
    )
    st.table(status_df)
