import pandas as pd
import requests
import time
import os
from tqdm import tqdm

# ======================
# Input / output files
# ======================

INPUT_FILE = "osm_round2_clean/osm_round2_clean_geocoding_input.csv"
OUTPUT_FILE = "osm_round2_clean/osm_round2_clean_geocoding_results.csv"

QUERY_COL = "osm_clean_query"

# Same User-Agent as previous OSM raw script
USER_AGENT = "msc_geocoding_project_xiaom/1.0"

# polite rate limit: 1 request per second
SLEEP_SECONDS = 1.1

# save every N records
SAVE_EVERY = 100


# ======================
# Geocoding function
# ======================

def geocode_osm(query):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            return {
                "osm_clean_success": False,
                "osm_clean_error": f"HTTP {response.status_code}",
                "osm_clean_lat": None,
                "osm_clean_lon": None,
                "osm_clean_display_name": None,
                "osm_clean_class": None,
                "osm_clean_type": None,
                "osm_clean_importance": None,
            }

        data = response.json()

        if len(data) == 0:
            return {
                "osm_clean_success": False,
                "osm_clean_error": "No result",
                "osm_clean_lat": None,
                "osm_clean_lon": None,
                "osm_clean_display_name": None,
                "osm_clean_class": None,
                "osm_clean_type": None,
                "osm_clean_importance": None,
            }

        result = data[0]

        return {
            "osm_clean_success": True,
            "osm_clean_error": None,
            "osm_clean_lat": result.get("lat"),
            "osm_clean_lon": result.get("lon"),
            "osm_clean_display_name": result.get("display_name"),
            "osm_clean_class": result.get("class"),
            "osm_clean_type": result.get("type"),
            "osm_clean_importance": result.get("importance"),
        }

    except Exception as e:
        return {
            "osm_clean_success": False,
            "osm_clean_error": str(e),
            "osm_clean_lat": None,
            "osm_clean_lon": None,
            "osm_clean_display_name": None,
            "osm_clean_class": None,
            "osm_clean_type": None,
            "osm_clean_importance": None,
        }


# ======================
# Load input
# ======================

df = pd.read_csv(INPUT_FILE)

if QUERY_COL not in df.columns:
    raise ValueError(f"Input file must contain a '{QUERY_COL}' column.")

print("Input records:", len(df), flush=True)


# ======================
# Resume if output exists
# ======================

if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)

    if QUERY_COL not in existing.columns:
        raise ValueError(f"Existing output file does not contain '{QUERY_COL}'.")

    done_queries = set(existing[QUERY_COL].dropna().astype(str))
    remaining = df[~df[QUERY_COL].astype(str).isin(done_queries)].copy()

    print("Existing results found:", len(existing), flush=True)
    print("Remaining queries:", len(remaining), flush=True)

    results = existing.to_dict("records")

else:
    remaining = df.copy()
    results = []

    print("No existing output found. Starting fresh.", flush=True)


# ======================
# Run geocoding
# ======================

for i, row in tqdm(
    remaining.iterrows(),
    total=len(remaining),
    desc="OSM round2 clean geocoding"
):
    query = str(row[QUERY_COL]).strip()

    base_record = row.to_dict()

    if query == "" or query.lower() in ["nan", "na", "none", "null"]:
        geo_result = {
            "osm_clean_success": False,
            "osm_clean_error": "Empty query",
            "osm_clean_lat": None,
            "osm_clean_lon": None,
            "osm_clean_display_name": None,
            "osm_clean_class": None,
            "osm_clean_type": None,
            "osm_clean_importance": None,
        }
    else:
        geo_result = geocode_osm(query)

    output_record = {**base_record, **geo_result}
    results.append(output_record)

    if len(results) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved progress: {len(results)} records", flush=True)

    time.sleep(SLEEP_SECONDS)


# ======================
# Final save and summary
# ======================

out = pd.DataFrame(results)
out.to_csv(OUTPUT_FILE, index=False)

success = out[out["osm_clean_success"] == True].copy()
failed = out[out["osm_clean_success"] == False].copy()

success.to_csv("osm_round2_clean/osm_round2_clean_success.csv", index=False)
failed.to_csv("osm_round2_clean/osm_round2_clean_failed.csv", index=False)

print("\n===== OSM ROUND2 CLEAN GEOCODING SUMMARY =====", flush=True)
print("Total queries:", len(out), flush=True)
print("Successful:", out["osm_clean_success"].sum(), flush=True)
print("Failed:", (~out["osm_clean_success"]).sum(), flush=True)
print("Success rate:", out["osm_clean_success"].mean(), flush=True)

print("\nSaved:", OUTPUT_FILE, flush=True)
