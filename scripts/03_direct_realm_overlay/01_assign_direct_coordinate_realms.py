#!/usr/bin/env python3

"""
Assign terrestrial realms to records that:

1. currently have no realm; and
2. have valid final coordinates.

The script does NOT rewrite the full master dataset. It produces a compact
assignment table that can be merged back after PTP and GBIF inference have
also been completed.

Only original coordinates are marked as eligible PTP realm evidence.
Site-geocoded and centroid-derived coordinates can receive a final realm,
but are not allowed to become PTP evidence.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

ID_COL = "BTseq_id"
REALM_COL = "realm"

FINAL_LAT_COL = "final_latitude"
FINAL_LON_COL = "final_longitude"

ORIGINAL_LAT_COL = "original_latitude"
ORIGINAL_LON_COL = "original_longitude"

COORDINATE_SOURCE_COL = "final_geocode_source"
COORDINATE_LEVEL_COL = "final_geocoding_level"

SHAPEFILE_REALM_COL = "REALM"

MISSING_TEXT_VALUES = {
    "",
    "nan",
    "none",
    "na",
    "n/a",
    "null",
    "<na>",
}


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def print_heading(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def text_missing(series: pd.Series) -> pd.Series:
    """Identify missing or blank text values."""
    cleaned = series.astype("string").str.strip().str.lower()

    return (
        series.isna()
        | cleaned.isna()
        | cleaned.isin(MISSING_TEXT_VALUES)
    )


def coordinate_valid(
    latitude: pd.Series,
    longitude: pd.Series,
) -> pd.Series:
    """
    Check whether latitude and longitude are numeric and fall within
    valid geographic ranges.
    """
    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")

    return (
        lat.notna()
        & lon.notna()
        & lat.between(-90, 90, inclusive="both")
        & lon.between(-180, 180, inclusive="both")
    )


def normalise_coordinate_source(series: pd.Series) -> pd.Series:
    """Standardise coordinate-source labels."""
    source = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    source = source.fillna("missing_source")
    source = source.replace(
        list(MISSING_TEXT_VALUES),
        "missing_source",
    )

    return source


def assignment_method_from_source(source: pd.Series) -> pd.Series:
    """Convert final coordinate source into a realm-assignment method."""
    method_map = {
        "original": "original_coordinate_overlay",
        "osm_raw": "site_geocoding_overlay",
        "osm_clean": "site_geocoding_overlay",
        "google_fallback": "site_geocoding_overlay",
        "province_centroid": "province_centroid_overlay",
        "country_centroid": "country_centroid_overlay",
        "missing_source": "unknown_coordinate_overlay",
    }

    return source.map(method_map).fillna(
        "unknown_coordinate_overlay"
    )


def confidence_from_source(source: pd.Series) -> pd.Series:
    """
    Assign a provenance-based confidence category.

    This confidence describes the coordinate source, not whether the
    polygon overlay itself succeeded.
    """
    confidence_map = {
        "original": "High",
        "osm_raw": "Medium",
        "osm_clean": "Medium",
        "google_fallback": "Medium",
        "province_centroid": "Low",
        "country_centroid": "Low",
        "missing_source": "Insufficient",
    }

    return source.map(confidence_map).fillna("Insufficient")


def collect_unique_realms(series: pd.Series) -> list[str]:
    """Collect sorted unique non-missing realm names."""
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values.ne("")]

    return sorted(set(values.tolist()))


# =============================================================================
# REALM POLYGON PREPARATION
# =============================================================================

def load_realm_polygons(shapefile: Path) -> gpd.GeoDataFrame:
    """Load and validate the RESOLVE realm polygons."""
    if not shapefile.exists():
        raise FileNotFoundError(
            f"Realm shapefile does not exist: {shapefile}"
        )

    realm_gdf = gpd.read_file(shapefile)

    if SHAPEFILE_REALM_COL not in realm_gdf.columns:
        raise KeyError(
            f"Realm column '{SHAPEFILE_REALM_COL}' not found.\n"
            f"Available columns: {realm_gdf.columns.tolist()}"
        )

    if realm_gdf.crs is None:
        raise ValueError("Realm shapefile has no CRS.")

    realm_gdf = realm_gdf[
        [SHAPEFILE_REALM_COL, "geometry"]
    ].copy()

    realm_gdf = realm_gdf[
        realm_gdf.geometry.notna()
        & ~realm_gdf.geometry.is_empty
    ].copy()

    realm_gdf[SHAPEFILE_REALM_COL] = (
        realm_gdf[SHAPEFILE_REALM_COL]
        .astype("string")
        .str.strip()
    )

    realm_gdf = realm_gdf[
        ~text_missing(realm_gdf[SHAPEFILE_REALM_COL])
    ].copy()

    invalid_geometry_n = int((~realm_gdf.geometry.is_valid).sum())

    if invalid_geometry_n:
        print(
            f"Repairing {invalid_geometry_n:,} invalid polygon geometries."
        )

        try:
            realm_gdf.geometry = realm_gdf.geometry.make_valid()
        except AttributeError:
            realm_gdf.geometry = realm_gdf.geometry.buffer(0)

    # Final coordinates are stored as longitude/latitude in EPSG:4326.
    realm_gdf = realm_gdf.to_crs("EPSG:4326")

    return realm_gdf


# =============================================================================
# SPATIAL OVERLAY
# =============================================================================

def assign_realms_to_chunk(
    target: pd.DataFrame,
    realm_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign realms to one chunk of target records."""
    points = gpd.GeoDataFrame(
        target.copy(),
        geometry=gpd.points_from_xy(
            target[FINAL_LON_COL],
            target[FINAL_LAT_COL],
        ),
        crs="EPSG:4326",
    )

    # Intersects is used so exact boundary points are detected.
    # Any point matching more than one distinct realm is retained as
    # a boundary conflict instead of receiving an arbitrary realm.
    joined = gpd.sjoin(
        points,
        realm_gdf,
        how="left",
        predicate="intersects",
    )

    overlay_summary = (
        joined.groupby(
            "input_row_number",
            sort=False,
            as_index=False,
        )
        .agg(
            matched_ecoregions_n=(
                "index_right",
                lambda x: int(x.notna().sum()),
            ),
            realm_candidates=(
                SHAPEFILE_REALM_COL,
                collect_unique_realms,
            ),
        )
    )

    overlay_summary["matched_realms_n"] = (
        overlay_summary["realm_candidates"].map(len)
    )

    overlay_summary["assigned_realm"] = (
        overlay_summary["realm_candidates"].map(
            lambda values: values[0] if len(values) == 1 else pd.NA
        )
    )

    overlay_summary["realm_candidate_list"] = (
        overlay_summary["realm_candidates"].map(
            lambda values: "|".join(values)
        )
    )

    overlay_summary["realm_overlay_status"] = np.select(
        [
            overlay_summary["matched_realms_n"].eq(1),
            overlay_summary["matched_realms_n"].gt(1),
        ],
        [
            "assigned",
            "boundary_or_overlap_conflict",
        ],
        default="no_terrestrial_realm",
    )

    overlay_summary = overlay_summary.drop(
        columns=["realm_candidates"]
    )

    result = target.merge(
        overlay_summary,
        on="input_row_number",
        how="left",
        validate="one_to_one",
    )

    return result


# =============================================================================
# CSV OUTPUT HELPERS
# =============================================================================

def append_csv(
    dataframe: pd.DataFrame,
    output_file: Path,
    first_write: bool,
) -> bool:
    """Write or append rows to a CSV file."""
    if dataframe.empty:
        return first_write

    dataframe.to_csv(
        output_file,
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )

    return False


def counter_to_rows(
    counter: Counter,
    summary_type: str,
    denominator: int | None = None,
) -> list[dict]:
    """Convert a Counter into summary-table rows."""
    rows = []

    for value, count in sorted(
        counter.items(),
        key=lambda item: (-item[1], str(item[0])),
    ):
        percentage = (
            round(count / denominator * 100, 4)
            if denominator
            else pd.NA
        )

        rows.append(
            {
                "summary_type": summary_type,
                "value": str(value),
                "records_n": int(count),
                "percentage": percentage,
            }
        )

    return rows


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def run(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    realm_shapefile = Path(args.realm_shapefile)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist: {input_csv}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_file = (
        output_dir
        / "direct_coordinate_realm_assignments.csv"
    )

    unassigned_file = (
        output_dir
        / "direct_coordinate_realm_unassigned.csv"
    )

    summary_file = (
        output_dir
        / "direct_coordinate_realm_assignment_summary.csv"
    )

    assignment_temp = assignment_file.with_suffix(".csv.tmp")
    unassigned_temp = unassigned_file.with_suffix(".csv.tmp")
    summary_temp = summary_file.with_suffix(".csv.tmp")

    # Remove incomplete temporary files from an earlier interrupted run.
    for temp_file in [
        assignment_temp,
        unassigned_temp,
        summary_temp,
    ]:
        if temp_file.exists():
            temp_file.unlink()

    print_heading("LOAD AND CHECK REALM POLYGONS")

    realm_gdf = load_realm_polygons(realm_shapefile)

    print(f"Realm polygons: {len(realm_gdf):,}")
    print(f"Realm CRS:      {realm_gdf.crs}")
    print("Realm values:")

    for value in sorted(
        realm_gdf[SHAPEFILE_REALM_COL]
        .dropna()
        .astype(str)
        .unique()
    ):
        print(f"  {value}")

    print_heading("CHECK INPUT COLUMNS")

    available_columns = pd.read_csv(
        input_csv,
        nrows=0,
    ).columns.tolist()

    required_columns = [
        ID_COL,
        REALM_COL,
        FINAL_LAT_COL,
        FINAL_LON_COL,
        ORIGINAL_LAT_COL,
        ORIGINAL_LON_COL,
        COORDINATE_SOURCE_COL,
        COORDINATE_LEVEL_COL,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in available_columns
    ]

    if missing_columns:
        raise KeyError(
            "Required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    print("Required columns found:")
    for col in required_columns:
        print(f"  {col}")

    print_heading("ASSIGN REALMS TO VALID FINAL COORDINATES")

    all_records_n = 0
    valid_final_coordinates_n = 0
    current_realm_missing_n = 0
    target_records_n = 0
    assigned_records_n = 0
    unassigned_records_n = 0
    ptp_evidence_eligible_n = 0

    source_counter = Counter()
    level_counter = Counter()
    method_counter = Counter()
    confidence_counter = Counter()
    overlay_status_counter = Counter()
    assigned_realm_counter = Counter()
    source_status_counter = Counter()

    assignment_first_write = True
    unassigned_first_write = True

    input_row_offset = 0

    reader = pd.read_csv(
        input_csv,
        usecols=required_columns,
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk_records_n = len(chunk)

        chunk["input_row_number"] = np.arange(
            input_row_offset,
            input_row_offset + chunk_records_n,
            dtype=np.int64,
        )

        input_row_offset += chunk_records_n
        all_records_n += chunk_records_n

        final_lat = pd.to_numeric(
            chunk[FINAL_LAT_COL],
            errors="coerce",
        )

        final_lon = pd.to_numeric(
            chunk[FINAL_LON_COL],
            errors="coerce",
        )

        original_lat = pd.to_numeric(
            chunk[ORIGINAL_LAT_COL],
            errors="coerce",
        )

        original_lon = pd.to_numeric(
            chunk[ORIGINAL_LON_COL],
            errors="coerce",
        )

        final_valid = coordinate_valid(
            final_lat,
            final_lon,
        )

        original_valid = coordinate_valid(
            original_lat,
            original_lon,
        )

        realm_missing = text_missing(chunk[REALM_COL])

        target_mask = final_valid & realm_missing

        valid_final_coordinates_n += int(final_valid.sum())
        current_realm_missing_n += int(realm_missing.sum())
        target_records_n += int(target_mask.sum())

        target = chunk.loc[
            target_mask,
            [
                "input_row_number",
                ID_COL,
                FINAL_LAT_COL,
                FINAL_LON_COL,
                ORIGINAL_LAT_COL,
                ORIGINAL_LON_COL,
                COORDINATE_SOURCE_COL,
                COORDINATE_LEVEL_COL,
            ],
        ].copy()

        if target.empty:
            print(
                f"Chunk {chunk_number:,}: "
                f"{chunk_records_n:,} rows | "
                "0 direct-overlay targets"
            )
            continue

        target[FINAL_LAT_COL] = final_lat.loc[target_mask].astype(float)
        target[FINAL_LON_COL] = final_lon.loc[target_mask].astype(float)

        target[ORIGINAL_LAT_COL] = original_lat.loc[target_mask]
        target[ORIGINAL_LON_COL] = original_lon.loc[target_mask]

        target["final_coordinate_source_standardised"] = (
            normalise_coordinate_source(
                target[COORDINATE_SOURCE_COL]
            )
        )

        target["realm_assignment_method"] = (
            assignment_method_from_source(
                target["final_coordinate_source_standardised"]
            )
        )

        target["realm_confidence"] = (
            confidence_from_source(
                target["final_coordinate_source_standardised"]
            )
        )

        result = assign_realms_to_chunk(
            target=target,
            realm_gdf=realm_gdf,
        )

        # Check that an "original" final coordinate is supported by a
        # valid original latitude/longitude pair.
        result_original_valid = coordinate_valid(
            result[ORIGINAL_LAT_COL],
            result[ORIGINAL_LON_COL],
        )

        original_final_match = (
            result_original_valid
            & np.isclose(
                pd.to_numeric(
                    result[ORIGINAL_LAT_COL],
                    errors="coerce",
                ),
                pd.to_numeric(
                    result[FINAL_LAT_COL],
                    errors="coerce",
                ),
                rtol=0,
                atol=1e-8,
            )
            & np.isclose(
                pd.to_numeric(
                    result[ORIGINAL_LON_COL],
                    errors="coerce",
                ),
                pd.to_numeric(
                    result[FINAL_LON_COL],
                    errors="coerce",
                ),
                rtol=0,
                atol=1e-8,
            )
        )

        result["original_final_coordinate_match"] = (
            original_final_match
        )

        result["ptp_evidence_eligible"] = (
            result["final_coordinate_source_standardised"].eq(
                "original"
            )
            & result["original_final_coordinate_match"]
            & result["realm_overlay_status"].eq("assigned")
        )

        assigned_mask = result["realm_overlay_status"].eq(
            "assigned"
        )

        assigned_records_n += int(assigned_mask.sum())
        unassigned_records_n += int((~assigned_mask).sum())

        ptp_evidence_eligible_n += int(
            result["ptp_evidence_eligible"].sum()
        )

        source_counter.update(
            result[
                "final_coordinate_source_standardised"
            ].fillna("missing_source")
        )

        level_counter.update(
            result[COORDINATE_LEVEL_COL]
            .fillna("missing_level")
            .astype(str)
        )

        method_counter.update(
            result["realm_assignment_method"]
        )

        confidence_counter.update(
            result["realm_confidence"]
        )

        overlay_status_counter.update(
            result["realm_overlay_status"]
        )

        assigned_realm_counter.update(
            result.loc[
                assigned_mask,
                "assigned_realm",
            ].dropna()
        )

        source_status_counter.update(
            zip(
                result[
                    "final_coordinate_source_standardised"
                ],
                result["realm_overlay_status"],
            )
        )

        output_columns = [
            "input_row_number",
            ID_COL,
            FINAL_LAT_COL,
            FINAL_LON_COL,
            COORDINATE_SOURCE_COL,
            "final_coordinate_source_standardised",
            COORDINATE_LEVEL_COL,
            "assigned_realm",
            "realm_assignment_method",
            "realm_confidence",
            "realm_overlay_status",
            "matched_ecoregions_n",
            "matched_realms_n",
            "realm_candidate_list",
            "original_final_coordinate_match",
            "ptp_evidence_eligible",
        ]

        assignment_first_write = append_csv(
            result[output_columns],
            assignment_temp,
            assignment_first_write,
        )

        unassigned_first_write = append_csv(
            result.loc[
                ~assigned_mask,
                output_columns,
            ],
            unassigned_temp,
            unassigned_first_write,
        )

        print(
            f"Chunk {chunk_number:,}: "
            f"{chunk_records_n:,} rows | "
            f"{len(target):,} targets | "
            f"{int(assigned_mask.sum()):,} assigned | "
            f"{int((~assigned_mask).sum()):,} unassigned"
        )

    # Ensure output files exist even if there are no matching records.
    output_columns = [
        "input_row_number",
        ID_COL,
        FINAL_LAT_COL,
        FINAL_LON_COL,
        COORDINATE_SOURCE_COL,
        "final_coordinate_source_standardised",
        COORDINATE_LEVEL_COL,
        "assigned_realm",
        "realm_assignment_method",
        "realm_confidence",
        "realm_overlay_status",
        "matched_ecoregions_n",
        "matched_realms_n",
        "realm_candidate_list",
        "original_final_coordinate_match",
        "ptp_evidence_eligible",
    ]

    if assignment_first_write:
        pd.DataFrame(columns=output_columns).to_csv(
            assignment_temp,
            index=False,
        )

    if unassigned_first_write:
        pd.DataFrame(columns=output_columns).to_csv(
            unassigned_temp,
            index=False,
        )

    summary_rows = [
        {
            "summary_type": "overall",
            "value": "all_records",
            "records_n": all_records_n,
            "percentage": 100.0,
        },
        {
            "summary_type": "overall",
            "value": "valid_final_coordinates",
            "records_n": valid_final_coordinates_n,
            "percentage": round(
                valid_final_coordinates_n
                / all_records_n
                * 100,
                4,
            ),
        },
        {
            "summary_type": "overall",
            "value": "current_realm_missing",
            "records_n": current_realm_missing_n,
            "percentage": round(
                current_realm_missing_n
                / all_records_n
                * 100,
                4,
            ),
        },
        {
            "summary_type": "overall",
            "value": "direct_overlay_targets",
            "records_n": target_records_n,
            "percentage": round(
                target_records_n
                / all_records_n
                * 100,
                4,
            ),
        },
        {
            "summary_type": "overall",
            "value": "realm_assigned",
            "records_n": assigned_records_n,
            "percentage": round(
                assigned_records_n
                / target_records_n
                * 100,
                4,
            ) if target_records_n else pd.NA,
        },
        {
            "summary_type": "overall",
            "value": "realm_unassigned",
            "records_n": unassigned_records_n,
            "percentage": round(
                unassigned_records_n
                / target_records_n
                * 100,
                4,
            ) if target_records_n else pd.NA,
        },
        {
            "summary_type": "overall",
            "value": "ptp_evidence_eligible",
            "records_n": ptp_evidence_eligible_n,
            "percentage": round(
                ptp_evidence_eligible_n
                / target_records_n
                * 100,
                4,
            ) if target_records_n else pd.NA,
        },
    ]

    summary_rows.extend(
        counter_to_rows(
            source_counter,
            "coordinate_source_among_targets",
            target_records_n,
        )
    )

    summary_rows.extend(
        counter_to_rows(
            level_counter,
            "coordinate_level_among_targets",
            target_records_n,
        )
    )

    summary_rows.extend(
        counter_to_rows(
            method_counter,
            "realm_assignment_method",
            target_records_n,
        )
    )

    summary_rows.extend(
        counter_to_rows(
            confidence_counter,
            "realm_confidence",
            target_records_n,
        )
    )

    summary_rows.extend(
        counter_to_rows(
            overlay_status_counter,
            "realm_overlay_status",
            target_records_n,
        )
    )

    summary_rows.extend(
        counter_to_rows(
            assigned_realm_counter,
            "assigned_realm",
            assigned_records_n,
        )
    )

    source_status_printable = Counter()

    for (source, status), count in source_status_counter.items():
        source_status_printable[
            f"{source} | {status}"
        ] = count

    summary_rows.extend(
        counter_to_rows(
            source_status_printable,
            "coordinate_source_by_overlay_status",
            target_records_n,
        )
    )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        summary_temp,
        index=False,
    )

    # Atomically replace final files only after successful completion.
    os.replace(assignment_temp, assignment_file)
    os.replace(unassigned_temp, unassigned_file)
    os.replace(summary_temp, summary_file)

    print_heading("FINAL SUMMARY")

    print(f"All records:                    {all_records_n:,}")
    print(
        f"Valid final coordinates:        "
        f"{valid_final_coordinates_n:,}"
    )
    print(
        f"Records with realm missing:     "
        f"{current_realm_missing_n:,}"
    )
    print(
        f"Direct-overlay targets:         "
        f"{target_records_n:,}"
    )
    print(
        f"Realm successfully assigned:    "
        f"{assigned_records_n:,}"
    )
    print(
        f"Realm not assigned:             "
        f"{unassigned_records_n:,}"
    )
    print(
        f"Eligible as PTP evidence:       "
        f"{ptp_evidence_eligible_n:,}"
    )

    if target_records_n != 161_283:
        print(
            "\nNOTE: The direct-overlay target count differs from "
            "the previously reported 161,283."
        )
        print(
            "Check whether the earlier inventory used exactly the "
            "same missing-value and coordinate-validity definitions."
        )

    print("\nSaved files:")
    print(assignment_file)
    print(unassigned_file)
    print(summary_file)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign RESOLVE terrestrial realms to records with "
            "valid final coordinates but missing realm."
        )
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default=(
            "centroid_results/"
            "new_all_seq_with_final_centroids.csv"
        ),
        help="Input master CSV.",
    )

    parser.add_argument(
        "--realm-shapefile",
        default=(
            "realm_data/resolve_ecoregions_2017/"
            "Ecoregions2017.shp"
        ),
        help="RESOLVE Ecoregions 2017 shapefile.",
    )

    parser.add_argument(
        "--output-dir",
        default="direct_coordinate_realm_assignment",
        help="Directory for compact assignment outputs.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Number of master-data rows processed per chunk.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
