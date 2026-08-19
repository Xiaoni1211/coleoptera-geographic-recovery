import pandas as pd
import requests
import time
import os
from tqdm import tqdm

# ======================
# Input / output files
# ======================

INPUT_FILE = "unique_precise_site_queries_raw.csv"
OUTPUT_FILE = "osm_raw_precise_geocoding_results.csv"

# Nominatim requires a clear User-Agent
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
                "osm_success": False,
                "osm_error": f"HTTP {response.status_code}",
                "osm_lat": None,
                "osm_lon": None,
                "osm_display_name": None,
                "osm_class": None,
                "osm_type": None,
                "osm_importance": None,
            }

        data = response.json()

        if len(data) == 0:
            return {
                "osm_success": False,
                "osm_error": "No result",
                "osm_lat": None,
                "osm_lon": None,
                "osm_display_name": None,
                "osm_class": None,
                "osm_type": None,
                "osm_importance": None,
            }

        result = data[0]

        return {
            "osm_success": True,
            "osm_error": None,
            "osm_lat": result.get("lat"),
            "osm_lon": result.get("lon"),
            "osm_display_name": result.get("display_name"),
            "osm_class": result.get("class"),
            "osm_type": result.get("type"),
            "osm_importance": result.get("importance"),
        }

    except Exception as e:
        return {
            "osm_success": False,
            "osm_error": str(e),
            "osm_lat": None,
            "osm_lon": None,
            "osm_display_name": None,
            "osm_class": None,
            "osm_type": None,
            "osm_importance": None,
        }


# ======================
# Load input
# ======================

df = pd.read_csv(INPUT_FILE)

if "raw_query" not in df.columns:
    raise ValueError("Input file must contain a 'raw_query' column.")

print("Input records:", len(df), flush=True)

# ======================
# Resume if output exists
# ======================

if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)

    if "raw_query" not in existing.columns:
        raise ValueError("Existing output file does not contain 'raw_query'.")

    done_queries = set(existing["raw_query"].dropna().astype(str))

    remaining = df[~df["raw_query"].astype(str).isin(done_queries)].copy()

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
    desc="OSM raw geocoding"
):
    query = str(row["raw_query"]).strip()

    base_record = row.to_dict()

    if query == "" or query.lower() in ["nan", "na", "none", "null"]:
        geo_result = {
            "osm_success": False,
            "osm_error": "Empty query",
            "osm_lat": None,
            "osm_lon": None,
            "osm_display_name": None,
            "osm_class": None,
            "osm_type": None,
            "osm_importance": None,
        }
    else:
        geo_result = geocode_osm(query)

    output_record = {**base_record, **geo_result}
    results.append(output_record)

    # save progress
    if len(results) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved progress: {len(results)} records", flush=True)

    time.sleep(SLEEP_SECONDS)


# ======================
# Final save and summary
# ======================

out = pd.DataFrame(results)
out.to_csv(OUTPUT_FILE, index=False)

print("\n===== OSM RAW GEOCODING SUMMARY =====", flush=True)
print("Total queries:", len(out), flush=True)
print("Successful:", out["osm_success"].sum(), flush=True)
print("Failed:", (~out["osm_success"]).sum(), flush=True)
print("Success rate:", out["osm_success"].mean(), flush=True)

print("\nSaved:", OUTPUT_FILE, flush=True)
