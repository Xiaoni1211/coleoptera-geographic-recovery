import pandas as pd
import numpy as np

OSM_RAW_FILE = "osm_raw_precise_geocoding_results.csv"
OSM_CLEAN_FILE = "osm_round2_clean/osm_round2_clean_geocoding_results.csv"
GOOGLE_FILE = "osm_round2_clean/google_round2_failed_results.csv"

OUT_FILE = "final_site_geocoding_lookup.csv"
OUT_SUMMARY = "final_site_geocoding_lookup_summary.csv"

def to_bool(x):
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def clean_query(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return x if x else np.nan

print("Loading OSM raw...")
raw = pd.read_csv(OSM_RAW_FILE, dtype=str, low_memory=False)
raw["raw_query"] = raw["raw_query"].apply(clean_query)
raw["osm_success_bool"] = raw["osm_success"].apply(to_bool)

print("Loading OSM clean...")
clean = pd.read_csv(OSM_CLEAN_FILE, dtype=str, low_memory=False)
clean["raw_query"] = clean["raw_query"].apply(clean_query)
clean["osm_clean_success_bool"] = clean["osm_clean_success"].apply(to_bool)

print("Loading Google fallback...")
google = pd.read_csv(GOOGLE_FILE, dtype=str, low_memory=False)
google["raw_query"] = google["raw_query"].apply(clean_query)
google["google_success_bool"] = google["google_success"].apply(to_bool)

# ============================================================
# Base: all raw OSM queries
# ============================================================

base_cols = [
    "country", "province_state", "region", "sector", "site", "raw_query",
    "osm_success", "osm_error", "osm_lat", "osm_lon",
    "osm_display_name", "osm_class", "osm_type", "osm_importance"
]

lookup = raw[base_cols].drop_duplicates(subset=["raw_query"]).copy()

# ============================================================
# Add OSM clean columns
# ============================================================

clean_cols = [
    "raw_query",
    "osm_clean_query", "osm_clean_success", "osm_clean_error",
    "osm_clean_lat", "osm_clean_lon",
    "osm_clean_display_name", "osm_clean_class",
    "osm_clean_type", "osm_clean_importance"
]

lookup = lookup.merge(
    clean[clean_cols].drop_duplicates(subset=["raw_query"]),
    on="raw_query",
    how="left"
)

# ============================================================
# Add Google columns
# ============================================================

google_cols = [
    "raw_query",
    "google_query", "google_success", "google_api_status", "google_error",
    "google_lat", "google_lon", "google_formatted_address",
    "google_place_id", "google_location_type",
    "google_partial_match", "google_types"
]

lookup = lookup.merge(
    google[google_cols].drop_duplicates(subset=["raw_query"]),
    on="raw_query",
    how="left"
)

# ============================================================
# Final coordinate decision
# Priority:
# 1. OSM raw success
# 2. OSM clean success
# 3. Google success
# ============================================================

lookup["osm_success_bool"] = lookup["osm_success"].apply(to_bool)
lookup["osm_clean_success_bool"] = lookup["osm_clean_success"].apply(to_bool)
lookup["google_success_bool"] = lookup["google_success"].apply(to_bool)

lookup["final_geocode_source"] = np.select(
    [
        lookup["osm_success_bool"],
        (~lookup["osm_success_bool"]) & lookup["osm_clean_success_bool"],
        (~lookup["osm_success_bool"]) & (~lookup["osm_clean_success_bool"]) & lookup["google_success_bool"],
    ],
    [
        "osm_raw",
        "osm_clean",
        "google_fallback",
    ],
    default="failed"
)

lookup["final_lat"] = np.select(
    [
        lookup["final_geocode_source"] == "osm_raw",
        lookup["final_geocode_source"] == "osm_clean",
        lookup["final_geocode_source"] == "google_fallback",
    ],
    [
        lookup["osm_lat"],
        lookup["osm_clean_lat"],
        lookup["google_lat"],
    ],
    default=np.nan
)

lookup["final_lon"] = np.select(
    [
        lookup["final_geocode_source"] == "osm_raw",
        lookup["final_geocode_source"] == "osm_clean",
        lookup["final_geocode_source"] == "google_fallback",
    ],
    [
        lookup["osm_lon"],
        lookup["osm_clean_lon"],
        lookup["google_lon"],
    ],
    default=np.nan
)

lookup["final_success"] = lookup["final_geocode_source"] != "failed"

# Metadata of chosen result
lookup["final_query_used"] = np.select(
    [
        lookup["final_geocode_source"] == "osm_raw",
        lookup["final_geocode_source"] == "osm_clean",
        lookup["final_geocode_source"] == "google_fallback",
    ],
    [
        lookup["raw_query"],
        lookup["osm_clean_query"],
        lookup["google_query"],
    ],
    default=np.nan
)

lookup["final_address_or_display_name"] = np.select(
    [
        lookup["final_geocode_source"] == "osm_raw",
        lookup["final_geocode_source"] == "osm_clean",
        lookup["final_geocode_source"] == "google_fallback",
    ],
    [
        lookup["osm_display_name"],
        lookup["osm_clean_display_name"],
        lookup["google_formatted_address"],
    ],
    default=np.nan
)

# ============================================================
# Summary
# ============================================================

summary = (
    lookup.groupby("final_geocode_source")
    .agg(
        unique_queries=("raw_query", "count")
    )
    .reset_index()
)

summary["percentage"] = (
    summary["unique_queries"] / len(lookup) * 100
).round(4)

print("\n===== Final geocoding lookup summary =====")
print(summary)

print("\nTotal unique raw queries:", len(lookup))
print("Final successful queries:", lookup["final_success"].sum())
print("Final failed queries:", (~lookup["final_success"]).sum())

# ============================================================
# Save
# ============================================================

lookup.to_csv(OUT_FILE, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

print("\nSaved:")
print(OUT_FILE)
print(OUT_SUMMARY)
print("\nDone.")
raw = pd.read_csv("osm_raw_precise_geocoding_results.csv")

print(raw.shape)

print(raw["raw_query"].isna().sum())

print(raw["raw_query"].duplicated().sum())

