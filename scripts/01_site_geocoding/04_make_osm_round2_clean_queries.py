import pandas as pd
import re
from pathlib import Path

# =========================
# 1. Settings
# =========================

INPUT_FILE = "osm_raw_precise_geocoding_results.csv"
OUTDIR = Path("osm_round2_clean")

SUCCESS_COL = "osm_success"
RAW_QUERY_COL = "raw_query"

OUTDIR.mkdir(exist_ok=True)

# =========================
# 2. Read OSM raw failed records
# =========================

df = pd.read_csv(INPUT_FILE)

failed = df[df[SUCCESS_COL] == False].copy()

print("=" * 60)
print(f"Total queries     : {len(df)}")
print(f"OSM raw success   : {df[SUCCESS_COL].sum()}")
print(f"OSM raw failed    : {len(failed)}")
print("=" * 60)

# =========================
# 3. Cleaning function
# =========================

def clean_query(q):
    if pd.isna(q):
        return ""

    q = str(q)

    # -------------------------
    # Standardise abbreviations
    # -------------------------
    replacements = {
        r"\bMt\.?\b": "Mount",
        r"\bMtn\.?\b": "Mountain",
        r"\bCk\.?\b": "Creek",
        r"\bCr\.?\b": "Creek",
        r"\bLk\.?\b": "Lake",
        r"\bR\.?\b": "River",
        r"\bJct\.?\b": "Junction",
        r"\bHwy\.?\b": "Highway",
        r"\bRd\.?\b": "Road",
        r"\bFt\.?\b": "Fort",
        r"\bGt\.?\b": "Great",
        r"\bNat\.?\b": "National",
        r"\bParq\.?\b": "Parque",
        r"\bProv\.?\b": "Provincial",
    }

    for pat, repl in replacements.items():
        q = re.sub(pat, repl, q, flags=re.IGNORECASE)

    # -------------------------
    # Remove approximation terms
    # -------------------------
    q = re.sub(
        r"\b(approx|approximately|ca\.?|circa|about)\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove distance + direction expressions
    # Examples:
    # 10 km north of
    # 5 miles SW of
    # 17 km SE
    # -------------------------
    q = re.sub(
        r"\b\d+(\.\d+)?\s*(km|kilometres?|kilometers?|miles?|mile|mi)\s*"
        r"(north|south|east|west|northeast|northwest|southeast|southwest|"
        r"n|s|e|w|ne|nw|se|sw)\s*(of|from)?\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove standalone distance expressions
    # Examples:
    # 10 km
    # 5 miles
    # -------------------------
    q = re.sub(
        r"\b\d+(\.\d+)?\s*(km|kilometres?|kilometers?|miles?|mile|mi)\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove directional phrases
    # Examples:
    # north of
    # south from
    # east side of
    # -------------------------
    q = re.sub(
        r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\s+"
        r"(of|from|side of|side|towards|toward)\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove single-letter direction only when isolated
    # Examples:
    # N of, SW of
    # -------------------------
    q = re.sub(
        r"\b(n|s|e|w|ne|nw|se|sw)\s+(of|from)\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove relative-position modifiers
    # Keep geographic names themselves
    # -------------------------
    q = re.sub(
        r"\b(near|around|vicinity of|close to|along|beside|opposite|towards|toward|"
        r"upstream of|downstream of|mouth of|entrance|gate|parking area|car park|"
        r"visitor centre|visitor center|campground|campsite)\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove sampling / collecting descriptors
    # -------------------------
    q = re.sub(
        r"\b(light trap|malaise trap|bait trap|trap|transect|quadrat|subplot|"
        r"sampling site|sample site|sample|sampling|collection site)\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove low-information field labels only when followed by codes
    # Examples:
    # Site A12, Plot 5, Station 3
    # -------------------------
    q = re.sub(
        r"\b(site|plot|station)\s*[A-Za-z]?\d+[A-Za-z\-]*\b",
        " ",
        q,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Clean punctuation
    # -------------------------
    q = re.sub(r"[;:/()\[\]{}]", " ", q)
    q = re.sub(r"\s*,\s*", ", ", q)
    q = re.sub(r"\s+", " ", q)

    # Remove repeated commas
    q = re.sub(r"(,\s*)+", ", ", q)
    q = q.strip(" ,.-")

    return q

# =========================
# 4. Apply cleaning
# =========================

failed["osm_clean_query"] = failed[RAW_QUERY_COL].apply(clean_query)

# =========================
# 5. Quality checks
# =========================

failed["raw_query_length"] = failed[RAW_QUERY_COL].astype(str).str.len()
failed["clean_query_length"] = failed["osm_clean_query"].astype(str).str.len()

failed["clean_query_empty"] = failed["osm_clean_query"].str.strip().eq("")
failed["clean_query_changed"] = failed[RAW_QUERY_COL] != failed["osm_clean_query"]

# avoid using empty cleaned queries
failed.loc[failed["clean_query_empty"], "osm_clean_query"] = failed.loc[
    failed["clean_query_empty"], RAW_QUERY_COL
]

# =========================
# 6. Export outputs
# =========================

failed.to_csv(
    OUTDIR / "osm_round2_clean_queries_full.csv",
    index=False
)

changed = failed[failed["clean_query_changed"]].copy()
unchanged = failed[~failed["clean_query_changed"]].copy()

changed.to_csv(
    OUTDIR / "osm_round2_clean_queries_changed.csv",
    index=False
)

unchanged.to_csv(
    OUTDIR / "osm_round2_clean_queries_unchanged.csv",
    index=False
)

# file for geocoding
geocode_input_cols = [
    col for col in [
        "country",
        "province_state",
        "region",
        "sector",
        "site",
        "raw_query",
        "osm_clean_query"
    ]
    if col in failed.columns
]

failed[geocode_input_cols].to_csv(
    OUTDIR / "osm_round2_clean_geocoding_input.csv",
    index=False
)

# sample for checking
failed[geocode_input_cols].sample(
    n=min(100, len(failed)),
    random_state=42
).to_csv(
    OUTDIR / "sample_osm_round2_clean_queries.csv",
    index=False
)

print("\n===== Round 2 clean query summary =====")
print(f"Failed records processed : {len(failed)}")
print(f"Changed queries          : {failed['clean_query_changed'].sum()}")
print(f"Unchanged queries        : {(~failed['clean_query_changed']).sum()}")
print(f"Empty cleaned queries    : {failed['clean_query_empty'].sum()}")

print("\nOutput folder:")
print(OUTDIR)

print("\nMain files:")
print("1. osm_round2_clean_queries_full.csv")
print("2. osm_round2_clean_geocoding_input.csv")
print("3. sample_osm_round2_clean_queries.csv")
