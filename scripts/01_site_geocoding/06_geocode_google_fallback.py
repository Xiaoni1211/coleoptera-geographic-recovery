import os
import time
import pandas as pd
import googlemaps

# =========================
# Settings
# =========================

INPUT_FILE = "osm_round2_clean_failed.csv"
OUTPUT_FILE = "google_round2_failed_results.csv"
CHECKPOINT_FILE = "google_round2_failed_results_checkpoint.csv"

QUERY_COL = "raw_query"   # use raw_query
SLEEP_SECONDS = 0.1

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if API_KEY is None:
    raise ValueError(
        "GOOGLE_MAPS_API_KEY is not set."
        "Please set it se an environment variable before running this project'"
    )

gmaps = googlemaps.Client(key=API_KEY)

# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("Google geocoding for Round 2 OSM failed records")
print("=" * 80)
print("Input file:", INPUT_FILE)
print("Input shape:", df.shape)
print("Using query column:", QUERY_COL)

if QUERY_COL not in df.columns:
    raise ValueError(f"{QUERY_COL} not found in input file.")

# =========================
# Resume from checkpoint
# =========================

if os.path.exists(CHECKPOINT_FILE):
    done = pd.read_csv(CHECKPOINT_FILE)
    print("Checkpoint found:", done.shape)

    processed_queries = set(done[QUERY_COL].astype(str))
    remaining = df[~df[QUERY_COL].astype(str).isin(processed_queries)].copy()

    results = done.to_dict("records")
    print("Remaining records:", remaining.shape[0])
else:
    remaining = df.copy()
    results = []
    print("No checkpoint found. Starting from beginning.")

# =========================
# Geocoding function
# =========================

def google_geocode(query):
    """
    Use Google Maps Geocoding API.

    Saves:
    - google_api_status
    - google_location_type
    - google_partial_match
    - google_types
    """

    try:
        response = gmaps.geocode(query)

        if len(response) == 0:
            return {
                "google_success": False,
                "google_api_status": "ZERO_RESULTS",
                "google_error": None,
                "google_lat": None,
                "google_lon": None,
                "google_formatted_address": None,
                "google_place_id": None,
                "google_location_type": None,
                "google_partial_match": None,
                "google_types": None,
            }

        r = response[0]
        geometry = r.get("geometry", {})
        loc = geometry.get("location", {})

        return {
            "google_success": True,
            "google_api_status": "OK",
            "google_error": None,
            "google_lat": loc.get("lat"),
            "google_lon": loc.get("lng"),
            "google_formatted_address": r.get("formatted_address"),
            "google_place_id": r.get("place_id"),
            "google_location_type": geometry.get("location_type"),
            "google_partial_match": r.get("partial_match", False),
            "google_types": "|".join(r.get("types", [])),
        }

    except googlemaps.exceptions.ApiError as e:
        return {
            "google_success": False,
            "google_api_status": str(e.status),
            "google_error": str(e),
            "google_lat": None,
            "google_lon": None,
            "google_formatted_address": None,
            "google_place_id": None,
            "google_location_type": None,
            "google_partial_match": None,
            "google_types": None,
        }

    except googlemaps.exceptions.TransportError as e:
        return {
            "google_success": False,
            "google_api_status": "TRANSPORT_ERROR",
            "google_error": str(e),
            "google_lat": None,
            "google_lon": None,
            "google_formatted_address": None,
            "google_place_id": None,
            "google_location_type": None,
            "google_partial_match": None,
            "google_types": None,
        }

    except googlemaps.exceptions.Timeout as e:
        return {
            "google_success": False,
            "google_api_status": "TIMEOUT",
            "google_error": str(e),
            "google_lat": None,
            "google_lon": None,
            "google_formatted_address": None,
            "google_place_id": None,
            "google_location_type": None,
            "google_partial_match": None,
            "google_types": None,
        }

    except Exception as e:
        return {
            "google_success": False,
            "google_api_status": "UNKNOWN_ERROR",
            "google_error": str(e),
            "google_lat": None,
            "google_lon": None,
            "google_formatted_address": None,
            "google_place_id": None,
            "google_location_type": None,
            "google_partial_match": None,
            "google_types": None,
        }

# =========================
# Run geocoding
# =========================

for i, row in remaining.iterrows():
    query = str(row[QUERY_COL])

    geo = google_geocode(query)

    out = row.to_dict()
    out["google_query"] = query
    out.update(geo)

    results.append(out)

    if len(results) % 100 == 0:
        temp = pd.DataFrame(results)
        temp.to_csv(CHECKPOINT_FILE, index=False)

        success_count = temp["google_success"].sum()
        print(
            f"Processed {len(results)} records | "
            f"Google success: {success_count} | "
            f"Success rate: {success_count / len(results):.3f}"
        )

    time.sleep(SLEEP_SECONDS)

# =========================
# Save final output
# =========================

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)
result_df.to_csv(CHECKPOINT_FILE, index=False)

print("=" * 80)
print("Google geocoding finished")
print("Output file:", OUTPUT_FILE)
print("Total records:", len(result_df))
print("Google success:", result_df["google_success"].sum())
print("Google failed:", (~result_df["google_success"]).sum())
print("Success rate:", result_df["google_success"].mean())

print("\nGoogle API status:")
print(result_df["google_api_status"].value_counts(dropna=False))

print("\nGoogle location type:")
print(result_df["google_location_type"].value_counts(dropna=False))

print("=" * 80)
