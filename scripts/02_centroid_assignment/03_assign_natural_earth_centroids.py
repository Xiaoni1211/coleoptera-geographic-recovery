#!/usr/bin/env python3
"""Match standardised centroid queries to Natural Earth and fill missing coordinates.

Run this script from ~/Msc:
    python assign_natural_earth_centroids.py

Required inputs:
    new_all_seq_with_site_geocoding.csv
    centroid_standardisation/standardised_unique_province_centroid_queries.csv
    centroid_standardisation/standardised_unique_country_centroid_queries.csv
    natural_earth/admin1/ne_10m_admin_1_states_provinces.shp
    natural_earth/admin0/ne_10m_admin_0_map_units.shp

The script never overwrites an existing valid final coordinate. Province centroids
are assigned first. Country centroids are assigned only to records without usable
province information, following the agreed hierarchy.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration: paths are relative to ~/Msc when the script is run there.
# -----------------------------------------------------------------------------
INPUT_DATA = Path("new_all_seq_with_site_geocoding.csv")
PROVINCE_QUERIES = Path(
    "centroid_standardisation/standardised_unique_province_centroid_queries.csv"
)
COUNTRY_QUERIES = Path(
    "centroid_standardisation/standardised_unique_country_centroid_queries.csv"
)
ADMIN1_SHP = Path("natural_earth/admin1/ne_10m_admin_1_states_provinces.shp")
ADMIN0_SHP = Path("natural_earth/admin0/ne_10m_admin_0_map_units.shp")

OUTPUT_DIR = Path("centroid_results")
FINAL_OUTPUT = OUTPUT_DIR / "new_all_seq_with_final_centroids.csv"
PROVINCE_LOOKUP_OUTPUT = OUTPUT_DIR / "province_centroid_lookup.csv"
COUNTRY_LOOKUP_OUTPUT = OUTPUT_DIR / "country_centroid_lookup.csv"
UNMATCHED_PROVINCE_OUTPUT = OUTPUT_DIR / "unmatched_province_centroid_queries.csv"
AMBIGUOUS_PROVINCE_OUTPUT = OUTPUT_DIR / "ambiguous_province_centroid_queries.csv"
UNMATCHED_COUNTRY_OUTPUT = OUTPUT_DIR / "unmatched_country_centroid_queries.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "centroid_assignment_summary.csv"

CHUNK_SIZE = 100_000
CENTROID_CRS = "EPSG:8857"  # Equal Earth; centroid is not calculated in EPSG:4326.

LAT_COL = "final_latitude"
LON_COL = "final_longitude"
SOURCE_COL = "final_geocode_source"
LEVEL_COL = "final_geocoding_level"

MISSING_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "unknown", "unrecoverable",
    "not available", "no data", "no locality", "not specified", "unspecified",
}


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input file(s) not found:\n  " + "\n  ".join(missing)
            + "\nRun the script from ~/Msc or correct the path constants at the top."
        )


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


def clean_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in MISSING_TOKENS else text


def normalise_name(value: object) -> str:
    """Build a stable, accent-insensitive administrative-name matching key."""
    text = clean_scalar(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_raw_series(series: pd.Series) -> pd.Series:
    """Normalise only formatting for merge-back keys; do not alter place meaning."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def valid_coordinate_mask(lat: pd.Series, lon: pd.Series) -> pd.Series:
    lat_num = pd.to_numeric(lat, errors="coerce")
    lon_num = pd.to_numeric(lon, errors="coerce")
    return lat_num.between(-90, 90) & lon_num.between(-180, 180)


def meaningful_series(series: pd.Series) -> pd.Series:
    value = normalise_raw_series(series)
    return ~value.isin(MISSING_TOKENS)


def first_valid_iso3(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype="object")
    for col in columns:
        if col not in frame.columns:
            continue
        candidate = frame[col].fillna("").astype(str).str.strip().str.upper()
        valid = candidate.str.fullmatch(r"[A-Z]{3}") & ~candidate.eq("-99")
        result = result.mask(result.eq("") & valid, candidate)
    return result


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        try:
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid].geometry.make_valid()
        except AttributeError:
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid].geometry.buffer(0)
    return gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def calculate_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return geometric, area-based centroids in WGS84 longitude/latitude."""
    if gdf.crs is None:
        raise ValueError("Natural Earth layer has no CRS information.")
    projected = gdf.to_crs(CENTROID_CRS)
    points = gpd.GeoSeries(projected.geometry.centroid, crs=CENTROID_CRS).to_crs(4326)
    result = gdf.copy()
    result["centroid_longitude"] = points.x.to_numpy()
    result["centroid_latitude"] = points.y.to_numpy()
    return result


def build_country_geometry() -> gpd.GeoDataFrame:
    banner("BUILDING NATURAL EARTH COUNTRY GEOMETRIES")
    admin0 = repair_geometries(gpd.read_file(ADMIN0_SHP))
    require_columns(admin0, ["ADM0_A3", "geometry"], "Natural Earth Admin-0")

    # ADM0_A3 is the principal key. ISO_A3/GU_A3 are safe fallbacks for rows
    # where Natural Earth stores -99 or a missing value.
    admin0["country_match_key"] = first_valid_iso3(
        admin0, ["ADM0_A3", "ISO_A3", "GU_A3"]
    )
    admin0 = admin0.loc[admin0["country_match_key"].ne("")].copy()

    country_names = (
        admin0.groupby("country_match_key", as_index=False)
        .agg(
            natural_earth_country=("ADMIN", lambda s: " | ".join(sorted(set(map(str, s.dropna()))))),
            natural_earth_map_units=("geometry", "size"),
        )
    )
    dissolved = admin0[["country_match_key", "geometry"]].dissolve(
        by="country_match_key", as_index=False
    )
    dissolved = dissolved.merge(country_names, on="country_match_key", how="left")
    dissolved = calculate_centroids(dissolved)
    print(f"Natural Earth country keys: {len(dissolved):,}")
    return dissolved


def build_admin1_geometry_and_aliases() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    banner("BUILDING NATURAL EARTH ADMIN-1 GEOMETRIES AND NAME ALIASES")
    admin1 = repair_geometries(gpd.read_file(ADMIN1_SHP))
    require_columns(admin1, ["adm1_code", "adm0_a3", "name", "geometry"], "Natural Earth Admin-1")

    admin1["adm0_a3"] = admin1["adm0_a3"].fillna("").astype(str).str.upper().str.strip()
    admin1["adm1_code"] = admin1["adm1_code"].fillna("").astype(str).str.strip()
    admin1 = admin1.loc[
        admin1["adm0_a3"].str.fullmatch(r"[A-Z]{3}") & admin1["adm1_code"].ne("")
    ].copy()

    metadata_columns = [c for c in ["adm1_code", "adm0_a3", "name", "name_en", "type_en"] if c in admin1]
    metadata = admin1[metadata_columns].drop_duplicates("adm1_code")
    dissolved = admin1[["adm1_code", "geometry"]].dissolve(by="adm1_code", as_index=False)
    dissolved = dissolved.merge(metadata, on="adm1_code", how="left")
    dissolved = calculate_centroids(dissolved)

    # Prefer English and principal names; alternative gazetteer names are fallbacks.
    alias_fields = ["name_en", "name", "gn_name", "gns_name", "name_alt", "name_local"]
    alias_frames: list[pd.DataFrame] = []
    for priority, field in enumerate(alias_fields, start=1):
        if field not in admin1.columns:
            continue
        part = admin1[["adm1_code", "adm0_a3", field]].copy()
        part["province_name_key"] = part[field].map(normalise_name)
        part = part.loc[part["province_name_key"].ne("")]
        part["alias_field"] = field
        part["alias_priority"] = priority
        alias_frames.append(
            part[["adm1_code", "adm0_a3", "province_name_key", "alias_field", "alias_priority"]]
        )

    aliases = pd.concat(alias_frames, ignore_index=True).drop_duplicates(
        ["adm1_code", "adm0_a3", "province_name_key"]
    )
    aliases["province_match_key"] = aliases["adm0_a3"] + "|" + aliases["province_name_key"]
    print(f"Natural Earth Admin-1 areas: {len(dissolved):,}")
    print(f"Natural Earth Admin-1 aliases: {len(aliases):,}")
    return dissolved, aliases


def match_country_queries(country_geometry: gpd.GeoDataFrame) -> pd.DataFrame:
    banner("MATCHING STANDARDISED COUNTRY QUERIES")
    queries = pd.read_csv(COUNTRY_QUERIES, low_memory=False)
    require_columns(
        queries,
        ["country", "country_iso", "country_match_key", "country_ready_for_natural_earth_matching"],
        "Standardised country queries",
    )
    queries["country_match_key"] = queries["country_match_key"].fillna("").astype(str).str.upper().str.strip()
    ready = queries["country_ready_for_natural_earth_matching"].astype(str).str.casefold().eq("true")

    geo = country_geometry.drop(columns="geometry").copy()
    lookup = queries.merge(geo, on="country_match_key", how="left", validate="many_to_one")
    lookup["country_centroid_match_status"] = np.select(
        [~ready, lookup["centroid_latitude"].notna()],
        ["not_ready_after_standardisation", "matched"],
        default="unmatched_natural_earth",
    )
    lookup.loc[~ready, ["centroid_latitude", "centroid_longitude"]] = np.nan
    lookup.to_csv(COUNTRY_LOOKUP_OUTPUT, index=False)
    lookup.loc[lookup["country_centroid_match_status"].ne("matched")].to_csv(
        UNMATCHED_COUNTRY_OUTPUT, index=False
    )
    print(lookup["country_centroid_match_status"].value_counts(dropna=False).to_string())
    return lookup


def match_province_queries(
    admin1_geometry: gpd.GeoDataFrame, aliases: pd.DataFrame
) -> pd.DataFrame:
    banner("MATCHING STANDARDISED PROVINCE QUERIES")
    queries = pd.read_csv(PROVINCE_QUERIES, low_memory=False)
    require_columns(
        queries,
        ["country", "country_iso", "province_state", "province_match_key", "province_standardisation_status"],
        "Standardised province queries",
    )
    queries = queries.reset_index(names="province_query_id")

    # Rebuild from the standardised ISO3 and standardised province key to ensure
    # exactly the same key structure is used on both sides.
    require_columns(
        queries,
        ["country_iso3_standardised", "province_state_normalised_key"],
        "Standardised province queries",
    )
    queries["_iso3"] = queries["country_iso3_standardised"].fillna("").astype(str).str.upper().str.strip()
    queries["_province_key"] = queries["province_state_normalised_key"].map(normalise_name)
    queries["_match_key"] = queries["_iso3"] + "|" + queries["_province_key"]

    candidates = queries[["province_query_id", "_match_key"]].merge(
        aliases,
        left_on="_match_key",
        right_on="province_match_key",
        how="left",
    )
    candidates = candidates.loc[candidates["adm1_code"].notna()].copy()

    # Keep only the highest-priority alias field available for each query.
    if not candidates.empty:
        best_priority = candidates.groupby("province_query_id")["alias_priority"].transform("min")
        candidates = candidates.loc[candidates["alias_priority"].eq(best_priority)].copy()
        candidates = candidates.drop_duplicates(["province_query_id", "adm1_code"])

    candidate_counts = candidates.groupby("province_query_id")["adm1_code"].nunique()
    unique_candidates = candidates.loc[
        candidates["province_query_id"].map(candidate_counts).eq(1)
    ].drop_duplicates("province_query_id")

    geom_cols = [
        "adm1_code", "adm0_a3", "name", "name_en", "type_en",
        "centroid_latitude", "centroid_longitude",
    ]
    geom_cols = [c for c in geom_cols if c in admin1_geometry.columns]
    matched = unique_candidates[["province_query_id", "adm1_code", "alias_field"]].merge(
        admin1_geometry.drop(columns="geometry")[geom_cols], on="adm1_code", how="left", validate="many_to_one"
    )
    lookup = queries.merge(matched, on="province_query_id", how="left", validate="one_to_one")
    lookup["natural_earth_candidate_count"] = (
        lookup["province_query_id"].map(candidate_counts).fillna(0).astype(int)
    )
    ready = lookup["province_standardisation_status"].eq("ready_for_natural_earth_matching")
    lookup["province_centroid_match_status"] = np.select(
        [
            ~ready,
            lookup["natural_earth_candidate_count"].gt(1),
            lookup["centroid_latitude"].notna(),
        ],
        ["not_ready_after_standardisation", "ambiguous_natural_earth_match", "matched"],
        default="unmatched_natural_earth",
    )
    lookup.loc[
        lookup["province_centroid_match_status"].ne("matched"),
        ["centroid_latitude", "centroid_longitude"],
    ] = np.nan
    lookup = lookup.drop(columns=["_iso3", "_province_key", "_match_key"])
    lookup.to_csv(PROVINCE_LOOKUP_OUTPUT, index=False)
    lookup.loc[lookup["province_centroid_match_status"].eq("unmatched_natural_earth")].to_csv(
        UNMATCHED_PROVINCE_OUTPUT, index=False
    )
    lookup.loc[lookup["province_centroid_match_status"].eq("ambiguous_natural_earth_match")].to_csv(
        AMBIGUOUS_PROVINCE_OUTPUT, index=False
    )
    print(lookup["province_centroid_match_status"].value_counts(dropna=False).to_string())
    return lookup


def prepare_merge_lookup(lookup: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "province":
        raw_columns = ["country", "country_iso", "province_state"]
        status_col = "province_centroid_match_status"
    else:
        raw_columns = ["country", "country_iso"]
        status_col = "country_centroid_match_status"

    matched = lookup.loc[
        lookup[status_col].eq("matched"),
        raw_columns + ["centroid_latitude", "centroid_longitude"],
    ].copy()
    merge_keys = []
    for col in raw_columns:
        key = f"_join_{col}"
        matched[key] = normalise_raw_series(matched[col])
        merge_keys.append(key)

    # A duplicate raw key is safe only when every duplicate has the same result.
    coordinate_counts = matched.groupby(merge_keys, dropna=False).agg(
        lat_n=("centroid_latitude", "nunique"), lon_n=("centroid_longitude", "nunique")
    )
    conflicts = coordinate_counts.loc[(coordinate_counts["lat_n"] > 1) | (coordinate_counts["lon_n"] > 1)]
    if not conflicts.empty:
        raise ValueError(
            f"Conflicting {level} centroid coordinates found for {len(conflicts)} raw merge key(s)."
        )
    return matched[merge_keys + ["centroid_latitude", "centroid_longitude"]].drop_duplicates(merge_keys)


def assign_to_main_data(province_lookup: pd.DataFrame, country_lookup: pd.DataFrame) -> dict[str, int]:
    banner("ASSIGNING CENTROIDS TO THE MAIN DATASET")
    province_merge = prepare_merge_lookup(province_lookup, "province").rename(
        columns={
            "centroid_latitude": "_province_centroid_latitude",
            "centroid_longitude": "_province_centroid_longitude",
        }
    )
    country_merge = prepare_merge_lookup(country_lookup, "country").rename(
        columns={
            "centroid_latitude": "_country_centroid_latitude",
            "centroid_longitude": "_country_centroid_longitude",
        }
    )

    if FINAL_OUTPUT.exists():
        FINAL_OUTPUT.unlink()

    totals = {
        "total_records": 0,
        "valid_coordinates_before_centroid": 0,
        "province_centroids_added": 0,
        "country_centroids_added": 0,
        "unresolved_after_centroid": 0,
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(INPUT_DATA, chunksize=CHUNK_SIZE, low_memory=False), start=1
    ):
        require_columns(
            chunk,
            ["country", "country_iso", "province_state", LAT_COL, LON_COL],
            "Main dataset",
        )
        if SOURCE_COL not in chunk.columns:
            chunk[SOURCE_COL] = ""
        if LEVEL_COL not in chunk.columns:
            chunk[LEVEL_COL] = ""

        chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")
        for col in ["country", "country_iso", "province_state"]:
            chunk[f"_join_{col}"] = normalise_raw_series(chunk[col])

        chunk = chunk.merge(
            province_merge,
            on=["_join_country", "_join_country_iso", "_join_province_state"],
            how="left",
            validate="many_to_one",
        )
        chunk = chunk.merge(
            country_merge,
            on=["_join_country", "_join_country_iso"],
            how="left",
            validate="many_to_one",
        )

        valid_before = valid_coordinate_mask(chunk[LAT_COL], chunk[LON_COL])
        province_available = (
            chunk["_province_centroid_latitude"].notna()
            & chunk["_province_centroid_longitude"].notna()
        )
        assign_province = ~valid_before & province_available

        chunk.loc[assign_province, LAT_COL] = chunk.loc[
            assign_province, "_province_centroid_latitude"
        ]
        chunk.loc[assign_province, LON_COL] = chunk.loc[
            assign_province, "_province_centroid_longitude"
        ]
        # These columns may be inferred as float64 when a chunk contains
        # only missing values. Convert them before assigning text labels.
        for text_col in (SOURCE_COL, LEVEL_COL):
            if text_col not in chunk.columns:
                chunk[text_col] = pd.Series(
                    pd.NA, index=chunk.index, dtype="object"
                )
            else:
                chunk[text_col] = chunk[text_col].astype("object")
        chunk.loc[assign_province, SOURCE_COL] = "province_centroid"
        chunk.loc[assign_province, LEVEL_COL] = "province"

        valid_after_province = valid_coordinate_mask(chunk[LAT_COL], chunk[LON_COL])
        no_usable_province = ~meaningful_series(chunk["province_state"])
        country_available = (
            chunk["_country_centroid_latitude"].notna()
            & chunk["_country_centroid_longitude"].notna()
        )
        assign_country = ~valid_after_province & no_usable_province & country_available

        chunk.loc[assign_country, LAT_COL] = chunk.loc[
            assign_country, "_country_centroid_latitude"
        ]
        chunk.loc[assign_country, LON_COL] = chunk.loc[
            assign_country, "_country_centroid_longitude"
        ]
        chunk.loc[assign_country, SOURCE_COL] = "country_centroid"
        chunk.loc[assign_country, LEVEL_COL] = "country"

        valid_final = valid_coordinate_mask(chunk[LAT_COL], chunk[LON_COL])
        totals["total_records"] += len(chunk)
        totals["valid_coordinates_before_centroid"] += int(valid_before.sum())
        totals["province_centroids_added"] += int(assign_province.sum())
        totals["country_centroids_added"] += int(assign_country.sum())
        totals["unresolved_after_centroid"] += int((~valid_final).sum())

        helper_columns = [c for c in chunk.columns if c.startswith("_join_")]
        helper_columns += [
            "_province_centroid_latitude", "_province_centroid_longitude",
            "_country_centroid_latitude", "_country_centroid_longitude",
        ]
        chunk = chunk.drop(columns=helper_columns)
        chunk.to_csv(
            FINAL_OUTPUT,
            mode="w" if chunk_number == 1 else "a",
            header=chunk_number == 1,
            index=False,
        )
        print(
            f"Chunk {chunk_number}: {len(chunk):,} rows | "
            f"province added {int(assign_province.sum()):,} | "
            f"country added {int(assign_country.sum()):,} | "
            f"unresolved {int((~valid_final).sum()):,}"
        )

    totals["valid_coordinates_after_centroid"] = (
        totals["total_records"] - totals["unresolved_after_centroid"]
    )
    return totals


def save_summary(
    totals: dict[str, int], province_lookup: pd.DataFrame, country_lookup: pd.DataFrame
) -> None:
    summary_rows = [{"metric": key, "value": value} for key, value in totals.items()]
    for status, count in province_lookup["province_centroid_match_status"].value_counts().items():
        summary_rows.append({"metric": f"province_queries_{status}", "value": int(count)})
    for status, count in country_lookup["country_centroid_match_status"].value_counts().items():
        summary_rows.append({"metric": f"country_queries_{status}", "value": int(count)})
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUTPUT, index=False)


def main() -> None:
    require_files([INPUT_DATA, PROVINCE_QUERIES, COUNTRY_QUERIES, ADMIN1_SHP, ADMIN0_SHP])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    country_geometry = build_country_geometry()
    admin1_geometry, admin1_aliases = build_admin1_geometry_and_aliases()
    country_lookup = match_country_queries(country_geometry)
    province_lookup = match_province_queries(admin1_geometry, admin1_aliases)
    totals = assign_to_main_data(province_lookup, country_lookup)
    save_summary(totals, province_lookup, country_lookup)

    banner("CENTROID ASSIGNMENT COMPLETE")
    for key, value in totals.items():
        print(f"{key}: {value:,}")
    print("\nOutputs:")
    for path in [
        FINAL_OUTPUT,
        PROVINCE_LOOKUP_OUTPUT,
        COUNTRY_LOOKUP_OUTPUT,
        UNMATCHED_PROVINCE_OUTPUT,
        AMBIGUOUS_PROVINCE_OUTPUT,
        UNMATCHED_COUNTRY_OUTPUT,
        SUMMARY_OUTPUT,
    ]:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
