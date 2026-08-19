#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge final OSM + Google site-level lookup back into the original dataset.

Input files
-----------
1. new_all_seq_standardised_tax_24042026_processed_clean_withptp_boldmeta 1.csv
2. final_site_geocoding_lookup_simple.csv

Main output
-----------
new_all_seq_with_site_geocoding.csv

Rules
-----
1. Keep valid original coordinates.
2. If original coordinates are missing/invalid, use successful site-level
   OSM/Google coordinates.
3. If neither is available, leave final coordinates missing for the later
   province/country centroid stage.
4. Merge using:
   country + province_state + region + sector + site
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# FILE PATHS
# =============================================================================

ORIGINAL_FILE = (
    "new_all_seq_standardised_tax_24042026_processed_clean_"
    "withptp_boldmeta 1.csv"
)

LOOKUP_FILE = "final_site_geocoding_lookup_simple.csv"

OUTPUT_FILE = "new_all_seq_with_site_geocoding.csv"
SUMMARY_FILE = "site_geocoding_merge_summary.csv"
UNMATCHED_FILE = "unmatched_site_geocoding_records.csv"
DUPLICATE_FILE = "duplicate_site_lookup_keys.csv"


# =============================================================================
# SETTINGS
# =============================================================================

MERGE_COLUMNS = [
    "country",
    "province_state",
    "region",
    "sector",
    "site",
]

CHUNK_SIZE = 100_000

# WKT POINT coordinates normally use:
# POINT (longitude latitude)
POINT_ORDER_IS_LON_LAT = True


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

def normalise_text(series):
    """
    Standardise text used for matching.

    Operations:
    - convert missing values to empty strings
    - remove leading/trailing spaces
    - collapse repeated spaces
    - convert to lowercase
    """
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def has_meaningful_text(series):
    """Return True for non-empty, meaningful text values."""
    text = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    missing_values = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "unknown",
        "not available",
        "no data",
    }

    return ~text.isin(missing_values)


def valid_coordinates(latitude, longitude):
    """
    Return a Boolean Series indicating valid coordinates.

    Valid coordinates must:
    - contain numeric latitude and longitude
    - have latitude between -90 and 90
    - have longitude between -180 and 180
    - not be exactly 0, 0
    """
    latitude = pd.to_numeric(latitude, errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")

    return (
        latitude.notna()
        & longitude.notna()
        & latitude.between(-90, 90)
        & longitude.between(-180, 180)
        & ~((latitude == 0) & (longitude == 0))
    )


def find_column(columns, possible_names, description):
    """Return the first matching column from possible_names."""
    for column in possible_names:
        if column in columns:
            return column

    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {possible_names}\n"
        f"Available columns: {list(columns)}"
    )


# =============================================================================
# ORIGINAL COORDINATE PARSING
# =============================================================================

def parse_coord(value):
    """
    Extract latitude and longitude from the original coord field.

    Supported examples:
        POINT (-1.234 52.345)
        POINT(-1.234 52.345)
        52.345, -1.234
        52.345 -1.234

    WKT POINT is interpreted as:
        POINT (longitude latitude)

    Other coordinate pairs are interpreted as:
        latitude, longitude

    If the first value is outside the possible latitude range but the
    second value is a possible latitude, the values are automatically
    interpreted as longitude, latitude.
    """
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "unknown",
    }:
        return np.nan, np.nan

    # -------------------------------------------------------------------------
    # WKT POINT format
    # -------------------------------------------------------------------------

    point_match = re.search(
        r"point\s*\(\s*"
        r"([-+]?\d+(?:\.\d+)?)"
        r"\s+"
        r"([-+]?\d+(?:\.\d+)?)"
        r"\s*\)",
        text,
        flags=re.IGNORECASE,
    )

    if point_match:
        first = float(point_match.group(1))
        second = float(point_match.group(2))

        if POINT_ORDER_IS_LON_LAT:
            longitude = first
            latitude = second
        else:
            latitude = first
            longitude = second

        if (
            -90 <= latitude <= 90
            and -180 <= longitude <= 180
            and not (latitude == 0 and longitude == 0)
        ):
            return latitude, longitude

        return np.nan, np.nan

    # -------------------------------------------------------------------------
    # Other coordinate formats
    # -------------------------------------------------------------------------

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        text,
    )

    if len(numbers) < 2:
        return np.nan, np.nan

    first = float(numbers[0])
    second = float(numbers[1])

    # Default for non-WKT coordinates: latitude, longitude
    latitude = first
    longitude = second

    # Automatically swap when the first value cannot be latitude.
    if abs(first) > 90 and abs(second) <= 90:
        longitude = first
        latitude = second

    if (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and not (latitude == 0 and longitude == 0)
    ):
        return latitude, longitude

    return np.nan, np.nan


def extract_original_coordinates(chunk):
    """
    Extract original latitude and longitude from the available columns.

    Priority:
    1. coord
    2. decimalLatitude + decimalLongitude
    3. latitude + longitude
    """
    if "coord" in chunk.columns:
        parsed = chunk["coord"].apply(parse_coord)

        latitude = pd.Series(
            [item[0] for item in parsed],
            index=chunk.index,
            dtype="float64",
        )

        longitude = pd.Series(
            [item[1] for item in parsed],
            index=chunk.index,
            dtype="float64",
        )

        return latitude, longitude

    if (
        "decimalLatitude" in chunk.columns
        and "decimalLongitude" in chunk.columns
    ):
        latitude = pd.to_numeric(
            chunk["decimalLatitude"],
            errors="coerce",
        )

        longitude = pd.to_numeric(
            chunk["decimalLongitude"],
            errors="coerce",
        )

        return latitude, longitude

    if (
        "latitude" in chunk.columns
        and "longitude" in chunk.columns
    ):
        latitude = pd.to_numeric(
            chunk["latitude"],
            errors="coerce",
        )

        longitude = pd.to_numeric(
            chunk["longitude"],
            errors="coerce",
        )

        return latitude, longitude

    raise ValueError(
        "Could not find original coordinate columns.\n"
        "Expected one of:\n"
        "1. coord\n"
        "2. decimalLatitude + decimalLongitude\n"
        "3. latitude + longitude"
    )


# =============================================================================
# LOOKUP PREPARATION
# =============================================================================

def prepare_lookup():
    """Load and prepare the final site-level lookup."""
    print("=" * 80)
    print("LOADING FINAL SITE-LEVEL LOOKUP")
    print("=" * 80)

    lookup = pd.read_csv(
        LOOKUP_FILE,
        low_memory=False,
    )

    print(f"Lookup rows loaded: {len(lookup):,}")
    print(f"Lookup columns: {list(lookup.columns)}")

    # -------------------------------------------------------------------------
    # Check merge columns
    # -------------------------------------------------------------------------

    missing_merge_columns = [
        column
        for column in MERGE_COLUMNS
        if column not in lookup.columns
    ]

    if missing_merge_columns:
        raise ValueError(
            "Lookup file is missing required merge columns: "
            f"{missing_merge_columns}"
        )

    # -------------------------------------------------------------------------
    # Identify coordinate columns
    # -------------------------------------------------------------------------

    latitude_column = find_column(
        lookup.columns,
        [
            "final_lat",
            "final_latitude",
            "geocoded_latitude",
            "geocode_latitude",
            "latitude",
            "lat",
        ],
        "lookup latitude column",
    )

    longitude_column = find_column(
        lookup.columns,
        [
            "final_lon",
            "final_longitude",
            "geocoded_longitude",
            "geocode_longitude",
            "longitude",
            "lon",
            "lng",
        ],
        "lookup longitude column",
    )

    print(f"Lookup latitude column:  {latitude_column}")
    print(f"Lookup longitude column: {longitude_column}")

    lookup["site_lookup_latitude"] = pd.to_numeric(
        lookup[latitude_column],
        errors="coerce",
    )

    lookup["site_lookup_longitude"] = pd.to_numeric(
        lookup[longitude_column],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Geocoding source
    # -------------------------------------------------------------------------

    if "final_geocode_source" in lookup.columns:
        lookup["site_lookup_source"] = (
            lookup["final_geocode_source"]
        )

    elif "geocode_source" in lookup.columns:
        lookup["site_lookup_source"] = (
            lookup["geocode_source"]
        )

    else:
        lookup["site_lookup_source"] = "site_geocoding"

    # -------------------------------------------------------------------------
    # Success status
    # -------------------------------------------------------------------------

    if "final_success" in lookup.columns:
        success_text = (
            lookup["final_success"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
        )

        reported_success = success_text.isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "success",
            ]
        )
    else:
        reported_success = pd.Series(
            True,
            index=lookup.index,
        )

    coordinate_valid = valid_coordinates(
        lookup["site_lookup_latitude"],
        lookup["site_lookup_longitude"],
    )

    lookup["site_lookup_success"] = (
        reported_success & coordinate_valid
    )

    # Failed lookup rows must not provide coordinates.
    failed_mask = ~lookup["site_lookup_success"]

    lookup.loc[
        failed_mask,
        "site_lookup_latitude",
    ] = np.nan

    lookup.loc[
        failed_mask,
        "site_lookup_longitude",
    ] = np.nan

    lookup.loc[
        failed_mask,
        "site_lookup_source",
    ] = pd.NA

    # -------------------------------------------------------------------------
    # Create normalised merge keys
    # -------------------------------------------------------------------------

    key_columns = []

    for column in MERGE_COLUMNS:
        key_column = f"_key_{column}"

        lookup[key_column] = normalise_text(
            lookup[column]
        )

        key_columns.append(key_column)

    # -------------------------------------------------------------------------
    # Check duplicated merge keys
    # -------------------------------------------------------------------------

    duplicate_mask = lookup.duplicated(
        subset=key_columns,
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        duplicate_rows = lookup.loc[
            duplicate_mask
        ].copy()

        duplicate_rows.to_csv(
            DUPLICATE_FILE,
            index=False,
        )

        print(
            f"Warning: {duplicate_count:,} lookup rows have "
            "duplicated merge keys."
        )

        print(
            f"They were saved to {DUPLICATE_FILE}."
        )

        # Prefer successful records.
        # If more than one successful record exists, prefer:
        # OSM raw > OSM clean > Google fallback.
        source_priority = {
            "osm_raw": 1,
            "osm_clean": 2,
            "google_fallback": 3,
            "failed": 9,
        }

        lookup["_source_priority"] = (
            lookup["site_lookup_source"]
            .map(source_priority)
            .fillna(8)
        )

        lookup = (
            lookup.sort_values(
                by=[
                    "site_lookup_success",
                    "_source_priority",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .drop_duplicates(
                subset=key_columns,
                keep="first",
            )
            .drop(columns="_source_priority")
            .reset_index(drop=True)
        )

    # Keep only the columns required for merging.
    lookup = lookup[
        key_columns
        + [
            "site_lookup_latitude",
            "site_lookup_longitude",
            "site_lookup_source",
            "site_lookup_success",
        ]
    ].copy()

    successful_count = int(
        lookup["site_lookup_success"].sum()
    )

    failed_count = len(lookup) - successful_count

    print(f"Unique lookup keys: {len(lookup):,}")
    print(f"Successful lookup keys: {successful_count:,}")
    print(f"Failed lookup keys: {failed_count:,}")

    return lookup, key_columns


# =============================================================================
# MAIN MERGE
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    if not Path(ORIGINAL_FILE).exists():
        raise FileNotFoundError(
            f"Original file not found: {ORIGINAL_FILE}"
        )

    if not Path(LOOKUP_FILE).exists():
        raise FileNotFoundError(
            f"Lookup file not found: {LOOKUP_FILE}"
        )

    lookup, key_columns = prepare_lookup()

    print()
    print("=" * 80)
    print("MERGING LOOKUP BACK INTO ORIGINAL DATA")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Summary counters
    # -------------------------------------------------------------------------

    total_rows = 0
    original_coordinate_rows = 0
    site_geocoded_rows = 0
    final_coordinate_rows = 0
    unresolved_rows = 0
    lookup_matched_rows = 0
    unmatched_site_candidate_rows = 0

    first_output_chunk = True
    first_unmatched_chunk = True

    original_reader = pd.read_csv(
        ORIGINAL_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        original_reader,
        start=1,
    ):
        input_chunk_rows = len(chunk)
        total_rows += input_chunk_rows

        # ---------------------------------------------------------------------
        # Check required columns
        # ---------------------------------------------------------------------

        missing_columns = [
            column
            for column in MERGE_COLUMNS
            if column not in chunk.columns
        ]

        if missing_columns:
            raise ValueError(
                "Original file is missing required merge columns: "
                f"{missing_columns}"
            )

        # ---------------------------------------------------------------------
        # Extract original coordinates
        # ---------------------------------------------------------------------

        original_latitude, original_longitude = (
            extract_original_coordinates(chunk)
        )

        chunk["original_latitude"] = (
            original_latitude.to_numpy()
        )

        chunk["original_longitude"] = (
            original_longitude.to_numpy()
        )

        original_valid_array = np.asarray(
            valid_coordinates(
                chunk["original_latitude"],
                chunk["original_longitude"],
            ),
            dtype=bool,
        )

        # ---------------------------------------------------------------------
        # Create normalised merge keys
        # ---------------------------------------------------------------------

        for column in MERGE_COLUMNS:
            chunk[f"_key_{column}"] = normalise_text(
                chunk[column]
            )

        # Preserve original order.
        chunk["_original_row_order"] = np.arange(
            input_chunk_rows
        )

        # ---------------------------------------------------------------------
        # Merge lookup into the current chunk
        # ---------------------------------------------------------------------

        merged = chunk.merge(
            lookup,
            how="left",
            on=key_columns,
            validate="many_to_one",
            sort=False,
        )

        merged = (
            merged.sort_values("_original_row_order")
            .reset_index(drop=True)
        )

        if len(merged) != input_chunk_rows:
            raise RuntimeError(
                "Row count changed during merge.\n"
                f"Before merge: {input_chunk_rows:,}\n"
                f"After merge:  {len(merged):,}"
            )

        # ---------------------------------------------------------------------
        # Determine available site-level coordinates
        # ---------------------------------------------------------------------

        site_lookup_valid_array = np.asarray(
            valid_coordinates(
                merged["site_lookup_latitude"],
                merged["site_lookup_longitude"],
            ),
            dtype=bool,
        )

        use_site_geocoding = (
            ~original_valid_array
            & site_lookup_valid_array
        )

        lookup_matched = (
            merged["site_lookup_success"]
            .notna()
            .to_numpy(dtype=bool)
        )

        # ---------------------------------------------------------------------
        # Create final latitude and longitude
        # ---------------------------------------------------------------------

        original_latitude_array = pd.to_numeric(
            merged["original_latitude"],
            errors="coerce",
        ).to_numpy()

        original_longitude_array = pd.to_numeric(
            merged["original_longitude"],
            errors="coerce",
        ).to_numpy()

        lookup_latitude_array = pd.to_numeric(
            merged["site_lookup_latitude"],
            errors="coerce",
        ).to_numpy()

        lookup_longitude_array = pd.to_numeric(
            merged["site_lookup_longitude"],
            errors="coerce",
        ).to_numpy()

        merged["final_latitude"] = np.where(
            original_valid_array,
            original_latitude_array,
            np.where(
                site_lookup_valid_array,
                lookup_latitude_array,
                np.nan,
            ),
        )

        merged["final_longitude"] = np.where(
            original_valid_array,
            original_longitude_array,
            np.where(
                site_lookup_valid_array,
                lookup_longitude_array,
                np.nan,
            ),
        )

        # ---------------------------------------------------------------------
        # Create final coordinate source
        # ---------------------------------------------------------------------

        final_source = pd.Series(
            pd.NA,
            index=merged.index,
            dtype="object",
        )

        original_positions = np.flatnonzero(
            original_valid_array
        )

        site_positions = np.flatnonzero(
            use_site_geocoding
        )

        final_source.iloc[
            original_positions
        ] = "original"

        if len(site_positions) > 0:
            final_source.iloc[
                site_positions
            ] = (
                merged["site_lookup_source"]
                .iloc[site_positions]
                .to_numpy()
            )

        merged["final_geocode_source"] = final_source

        # ---------------------------------------------------------------------
        # Create final geocoding level
        # ---------------------------------------------------------------------

        final_level = pd.Series(
            pd.NA,
            index=merged.index,
            dtype="object",
        )

        final_level.iloc[
            original_positions
        ] = "original"

        final_level.iloc[
            site_positions
        ] = "site"

        merged["final_geocoding_level"] = final_level

        # ---------------------------------------------------------------------
        # Final coordinate status
        # ---------------------------------------------------------------------

        final_valid_array = np.asarray(
            valid_coordinates(
                merged["final_latitude"],
                merged["final_longitude"],
            ),
            dtype=bool,
        )

        merged["final_coordinate_valid"] = (
            final_valid_array
        )

        merged["site_geocoding_added"] = (
            use_site_geocoding
        )

        merged["site_lookup_matched"] = (
            lookup_matched
        )

        # ---------------------------------------------------------------------
        # Classify the next centroid requirement
        # ---------------------------------------------------------------------

        country_present = (
            has_meaningful_text(merged["country"])
            .to_numpy(dtype=bool)
        )

        province_present = (
            has_meaningful_text(
                merged["province_state"]
            )
            .to_numpy(dtype=bool)
        )

        still_missing = ~final_valid_array

        centroid_requirement = np.full(
            len(merged),
            "not_needed",
            dtype=object,
        )

        centroid_requirement[
            still_missing
            & country_present
            & province_present
        ] = "province_centroid"

        centroid_requirement[
            still_missing
            & country_present
            & ~province_present
        ] = "country_centroid"

        centroid_requirement[
            still_missing
            & ~country_present
        ] = "no_geographic_information"

        merged["centroid_requirement"] = (
            centroid_requirement
        )

        # ---------------------------------------------------------------------
        # Identify unresolved records with site and country information
        # ---------------------------------------------------------------------

        site_present = (
            has_meaningful_text(merged["site"])
            .to_numpy(dtype=bool)
        )

        unmatched_site_candidate = (
            still_missing
            & site_present
            & country_present
        )

        unmatched_positions = np.flatnonzero(
            unmatched_site_candidate
        )

        unmatched_output = merged.iloc[
            unmatched_positions
        ].copy()

        # ---------------------------------------------------------------------
        # Update totals
        # ---------------------------------------------------------------------

        original_count = int(
            original_valid_array.sum()
        )

        site_added_count = int(
            use_site_geocoding.sum()
        )

        final_valid_count = int(
            final_valid_array.sum()
        )

        unresolved_count = int(
            still_missing.sum()
        )

        lookup_matched_count = int(
            lookup_matched.sum()
        )

        unmatched_site_count = int(
            unmatched_site_candidate.sum()
        )

        original_coordinate_rows += original_count
        site_geocoded_rows += site_added_count
        final_coordinate_rows += final_valid_count
        unresolved_rows += unresolved_count
        lookup_matched_rows += lookup_matched_count
        unmatched_site_candidate_rows += unmatched_site_count

        print(
            f"Chunk {chunk_number}: "
            f"{input_chunk_rows:,} rows | "
            f"original coordinates {original_count:,} | "
            f"site coordinates added {site_added_count:,} | "
            f"still unresolved {unresolved_count:,}"
        )

        # ---------------------------------------------------------------------
        # Remove temporary lookup columns
        # ---------------------------------------------------------------------

        temporary_columns = (
            key_columns
            + [
                "_original_row_order",
                "site_lookup_latitude",
                "site_lookup_longitude",
                "site_lookup_source",
                "site_lookup_success",
            ]
        )

        merged.drop(
            columns=[
                column
                for column in temporary_columns
                if column in merged.columns
            ],
            inplace=True,
        )

        unmatched_output.drop(
            columns=[
                column
                for column in temporary_columns
                if column in unmatched_output.columns
            ],
            inplace=True,
        )

        # ---------------------------------------------------------------------
        # Save main output
        # ---------------------------------------------------------------------

        merged.to_csv(
            OUTPUT_FILE,
            mode="w" if first_output_chunk else "a",
            header=first_output_chunk,
            index=False,
        )

        first_output_chunk = False

        # ---------------------------------------------------------------------
        # Save unresolved site records
        # ---------------------------------------------------------------------

        if not unmatched_output.empty:
            unmatched_output.to_csv(
                UNMATCHED_FILE,
                mode=(
                    "w"
                    if first_unmatched_chunk
                    else "a"
                ),
                header=first_unmatched_chunk,
                index=False,
            )

            first_unmatched_chunk = False

    # -------------------------------------------------------------------------
    # Ensure unmatched output is not left over from an earlier run
    # -------------------------------------------------------------------------

    if first_unmatched_chunk:
        pd.DataFrame(
            columns=[
                "message"
            ]
        ).assign(
            message=[
                "No unresolved site-level records found."
            ]
        ).to_csv(
            UNMATCHED_FILE,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    if (
        original_coordinate_rows
        + site_geocoded_rows
        != final_coordinate_rows
    ):
        raise RuntimeError(
            "Final coordinate count validation failed.\n"
            f"Original valid coordinates: "
            f"{original_coordinate_rows:,}\n"
            f"Site coordinates added: "
            f"{site_geocoded_rows:,}\n"
            f"Final valid coordinates: "
            f"{final_coordinate_rows:,}"
        )

    if final_coordinate_rows + unresolved_rows != total_rows:
        raise RuntimeError(
            "Final row classification validation failed."
        )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        {
            "category": [
                "total_records",
                "original_valid_coordinates",
                "site_geocoding_added",
                "records_with_final_coordinates",
                "still_missing_coordinates",
                "records_matched_to_lookup_key",
                "unresolved_site_candidates",
            ],
            "count": [
                total_rows,
                original_coordinate_rows,
                site_geocoded_rows,
                final_coordinate_rows,
                unresolved_rows,
                lookup_matched_rows,
                unmatched_site_candidate_rows,
            ],
        }
    )

    summary["percentage_of_total"] = (
        summary["count"]
        / total_rows
        * 100
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL SITE-LEVEL MERGE SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            formatters={
                "percentage_of_total": (
                    lambda value: f"{value:.4f}"
                )
            },
        )
    )

    print()
    print("Saved files:")
    print(f"1. {OUTPUT_FILE}")
    print(f"2. {SUMMARY_FILE}")
    print(f"3. {UNMATCHED_FILE}")

    if Path(DUPLICATE_FILE).exists():
        print(f"4. {DUPLICATE_FILE}")

    print()
    print(
        "Site-level lookup was merged successfully."
    )

    print(
        "Use the centroid_requirement column to identify "
        "the next processing step."
    )


if __name__ == "__main__":
    main()
