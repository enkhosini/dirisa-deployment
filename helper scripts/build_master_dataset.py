"""
Consolidates the 12 processed CSVs into a single, clean, municipality-level
master dataset for the DIRISA SDC 2026 Eastern Cape pilot.

Inputs used (municipality-grain, 33 rows each):
  - municipality_features.csv     base table: population, 2016 & 2021 election
                                   results, youth registration, risk score
  - clean_youth_registration.csv  adds `province`
  - municipality_clusters_full.csv adds `cluster` / `cluster_name` (all 33,
                                   including "Insufficient data")
  - priority_municipalities.csv   adds `priority_rank`
  - municipality_widest_gap.csv   adds `gap_rank`
  - clean_provincial_turnout_2021.csv  province-level 2021 turnout benchmark
                                   (Eastern Cape row) -> adds a constant
                                   reference column + a per-municipality
                                   delta against it

Inputs deliberately NOT merged in (different grain / redundant / stale):
  - clean_master_municipality.csv     an earlier-stage subset of
                                       municipality_features.csv (pre youth
                                       merge) - fully superseded
  - municipality_clusters.csv         only the 24 clustered rows -
                                       superseded by municipality_clusters_full.csv
  - interactive_export_priority.csv   a dashboard export view, same
                                       municipality-level fields already covered
  - clean_census_population.csv       district-level (6 rows), not
                                       municipality-level - can't merge 1:1
  - clean_district_youth_history.csv  district-level, 2 years x 6 districts -
                                       already summarised into
                                       `youth_registration_change_pct` via
                                       district_trend.csv / municipality_features.csv
  - district_trend.csv                district-level intermediate -
                                       already folded into municipality_features.csv

Output: municipality_master_dataset.csv (33 rows, one per municipality)
"""

import pandas as pd

# ---- Base table -----------------------------------------------------------
mf = pd.read_csv("municipality_features.csv")

# district_normalised is an exact duplicate of district; district_municipality
# and trend_rank are stray partial columns from an intermediate merge step
# (only 13/33 populated) that add nothing not already captured in
# youth_registration_change_pct / youth_disengagement_risk.
mf = mf.drop(columns=["district_normalised", "district_municipality", "trend_rank"])

# ---- Province ---------------------------------------------------------------
cy = pd.read_csv("clean_youth_registration.csv")[["mdb_code", "province"]]
df = mf.merge(cy, on="mdb_code", how="left")

# ---- Cluster assignment (all 33, incl. "Insufficient data") ----------------
clusters = pd.read_csv("municipality_clusters_full.csv")
cluster_lookup = clusters[["mdb_code", "cluster_name"]]
# municipality_clusters_full.csv doesn't carry the numeric id for the
# "Insufficient data" rows; pull the numeric cluster id from the 24-row file
# where it exists, leave blank otherwise.
cluster_ids = pd.read_csv("municipality_clusters.csv")[["mdb_code", "cluster"]]
cluster_lookup = cluster_lookup.merge(cluster_ids, on="mdb_code", how="left")
df = df.merge(cluster_lookup, on="mdb_code", how="left")

# ---- Priority rank & gap rank -----------------------------------------------
priority = pd.read_csv("priority_municipalities.csv")[["mdb_code", "priority_rank"]]
df = df.merge(priority, on="mdb_code", how="left")

gap = pd.read_csv("municipality_widest_gap.csv")[["mdb_code", "gap_rank"]]
df = df.merge(gap, on="mdb_code", how="left")

# ---- Provincial 2021 turnout benchmark --------------------------------------
prov_raw = pd.read_csv("clean_provincial_turnout_2021.csv")
# Row layout is a raw scrape: real header is in row index 2, Eastern Cape is
# row index 3 (see script docstring / inspection). % turnout column has a
# trailing '%' and a comma decimal separator.
ec_row = prov_raw.iloc[3]  # "Eastern Cape" data row (rows 0-2 are scrape preamble/header text)
ec_turnout_str = str(ec_row["Unnamed: 12"]).strip().replace("%", "").replace(",", ".")
provincial_turnout_2021 = float(ec_turnout_str)

df["provincial_turnout_pct_2021"] = provincial_turnout_2021
df["turnout_vs_provincial_avg_2021_pp"] = df["turnout_pct_2021"] - provincial_turnout_2021

# ---- Final column order -----------------------------------------------------
ordered_cols = [
    # identifiers
    "mdb_code", "municipality", "district", "province",
    # population / youth
    "total_pop_2022", "youth_pop_20_34_2022", "youth_share_pct_2022",
    # 2016 election
    "registered_2016", "valid_votes_2016", "spoilt_votes_2016", "votes_cast_2016",
    "turnout_pct_2016", "spoilt_rate_pct_2016",
    # 2021 election
    "registered_2021", "valid_votes_2021", "spoilt_votes_2021", "votes_cast_2021",
    "turnout_pct_2021", "spoilt_rate_pct_2021",
    # 2016->2021 change
    "registered_growth_2016_2021_pct", "turnout_change_2016_2021_pp",
    # provincial benchmark
    "provincial_turnout_pct_2021", "turnout_vs_provincial_avg_2021_pp",
    # youth registration / gap
    "registered_youth_total", "youth_gap_2026", "youth_registration_rate_2026",
    "youth_registration_change_pct",
    # model outputs
    "gap_norm", "trend_norm", "youth_disengagement_risk",
    "cluster", "cluster_name", "priority_rank", "gap_rank",
]
missing = set(df.columns) - set(ordered_cols)
assert not missing, f"Unplaced columns: {missing}"
df = df[ordered_cols].sort_values("priority_rank")

df.to_csv("municipality_master_dataset.csv", index=False)
print(f"Saved municipality_master_dataset.csv: {df.shape}")
print(df.head(5).to_string())
