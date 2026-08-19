#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Screen centroid candidates after site-level geocoding has been merged.

Run:
    python screen_centroid_candidates.py new_all_seq_with_site_geocoding.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_SIZE = 100_000

OUTPUT_PROVINCE = "province_centroid_candidates_after_site.csv"
OUTPUT_COUNTRY = "country_centroid_candidates_after_site.csv"
OUTPUT_UNRESOLVED_PROVINCE = "province_without_country_identifier.csv"

OUTPUT_UNIQUE_PROVINCE = "unique_province_centroid_queries.csv"
OUTPUT_UNIQUE_COUNTRY = "unique_country_centroid_queries.csv"

OUTPUT_SUMMARY = "centroid_screening_summary.csv"


COORDINATE_OPTIONS = [
    ("final_latitude", "final_longitude"),
    ("final_lat", "final_lon"),
    ("best_latitude", "best_longitude"),
]


MISSING_TEXT = {
    "",
    "nan",
    "none",
    "null",
    "unknown",
    "missing",
    "not available",
    "not provided",
    "no data",
    "n/a",
    "na",
}


def has_text(series, preserve_na=False):
    """Return True for fields containing usable text."""
    values = series.astype(str).str.strip().str.lower()

    missing_values = MISSING_TEXT.copy()

    # Namibia's ISO-2 code is NA, so do not treat NA as missing
    # when checking country_iso.
    if preserve_na:
        missing_values.discard("na")
        missing_values.discard("n/a")

    return ~values.isin(missing_values)


def detect_coordinate_columns(columns):
    for latitude_col, longitude_col in COORDINATE_OPTIONS:
        if latitude_col in columns and longitude_col in columns:
            return latitude_col, longitude_col

    raise ValueError(
        "Final coordinate columns were not found. Expected one of:\n"
        "final_latitude + final_longitude\n"
        "final_lat + final_lon\n"
        "best_latitude + best_longitude"
    )


def valid_coordinates(latitude, longitude):
    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")

    valid = (
        lat.notna()
        & lon.notna()
        & lat.between(-90, 90)
        & lon.between(-180, 180)
    )

    # Treat 0,0 as a placeholder rather than a valid location.
    zero_zero = lat.eq(0) & lon.eq(0)

    return valid & ~zero_zero


def initialise_csv(filename, columns):
    pd.DataFrame(columns=columns).to_csv(filename, index=False)


def create_unique_queries(
    candidate_file,
    output_file,
    grouping_columns,
):
    candidates = pd.read_csv(
        candidate_file,
        keep_default_na=False,
        low_memory=False,
    )

    if len(candidates) == 0:
        pd.DataFrame(
            columns=grouping_columns + ["record_count"]
        ).to_csv(output_file, index=False)
        return 0

    unique_queries = (
        candidates.groupby(
            grouping_columns,
            dropna=False,
        )
        .size()
        .reset_index(name="record_count")
        .sort_values(
            "record_count",
            ascending=False,
        )
    )

    unique_queries.to_csv(output_file, index=False)

    return len(unique_queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Complete CSV after site-level geocoding merge",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    header = pd.read_csv(
        input_path,
        nrows=0,
        keep_default_na=False,
    )

    columns = list(header.columns)

    for required_col in ["country", "province_state"]:
        if required_col not in columns:
            raise ValueError(
                f"Required column missing: {required_col}"
            )

    latitude_col, longitude_col = detect_coordinate_columns(columns)

    has_country_iso_column = "country_iso" in columns

    # Keep only useful record identifiers if they exist.
    possible_id_columns = [
        "BTseq_id",
        "db_id",
        "specimen_id",
        "record_id",
        "sampleid",
    ]

    id_columns = [
        col for col in possible_id_columns
        if col in columns
    ]

    optional_location_columns = [
        col for col in [
            "region",
            "sector",
            "site",
            "site_code",
        ]
        if col in columns
    ]

    read_columns = (
        id_columns
        + ["country", "province_state"]
        + optional_location_columns
        + [latitude_col, longitude_col]
    )

    if has_country_iso_column:
        read_columns.append("country_iso")

    # Remove any accidental duplicate column names.
    read_columns = list(dict.fromkeys(read_columns))

    output_columns = (
        ["source_data_row_number"]
        + id_columns
        + ["country", "country_iso", "province_state"]
        + optional_location_columns
        + [
            "centroid_target_level",
            "country_identifier_source",
        ]
    )

    initialise_csv(OUTPUT_PROVINCE, output_columns)
    initialise_csv(OUTPUT_COUNTRY, output_columns)
    initialise_csv(OUTPUT_UNRESOLVED_PROVINCE, output_columns)

    totals = {
        "total_records": 0,
        "valid_coordinate_records": 0,
        "missing_coordinate_records": 0,
        "province_centroid_candidates": 0,
        "country_centroid_candidates": 0,
        "province_without_country_or_iso": 0,
        "no_province_country_or_iso": 0,
    }

    processed_rows = 0

    print("=" * 80)
    print("CENTROID CANDIDATE SCREENING")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Latitude column:  {latitude_col}")
    print(f"Longitude column: {longitude_col}")
    print(f"country_iso available: {has_country_iso_column}")

    reader = pd.read_csv(
        input_path,
        usecols=read_columns,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk_size = len(chunk)

        # 1-based position within the data rows, excluding the header.
        chunk["source_data_row_number"] = (
            np.arange(
                processed_rows + 1,
                processed_rows + chunk_size + 1,
            )
        )

        if not has_country_iso_column:
            chunk["country_iso"] = ""

        has_coord = valid_coordinates(
            chunk[latitude_col],
            chunk[longitude_col],
        )

        missing_coord = ~has_coord

        has_country = has_text(chunk["country"])
        has_province = has_text(chunk["province_state"])

        # Preserve NA because it can mean Namibia.
        has_country_iso = has_text(
            chunk["country_iso"],
            preserve_na=True,
        )

        has_country_identifier = has_country | has_country_iso

        province_mask = (
            missing_coord
            & has_province
            & has_country_identifier
        )

        country_mask = (
            missing_coord
            & ~has_province
            & has_country_identifier
        )

        unresolved_province_mask = (
            missing_coord
            & has_province
            & ~has_country_identifier
        )

        no_admin_mask = (
            missing_coord
            & ~has_province
            & ~has_country_identifier
        )

        country_source = pd.Series(
            "",
            index=chunk.index,
            dtype="object",
        )

        country_source.loc[has_country] = "country"
        country_source.loc[
            ~has_country & has_country_iso
        ] = "country_iso"

        base_columns = (
            ["source_data_row_number"]
            + id_columns
            + ["country", "country_iso", "province_state"]
            + optional_location_columns
        )

        province_output = chunk.loc[
            province_mask,
            base_columns,
        ].copy()

        province_output["centroid_target_level"] = "province"
        province_output["country_identifier_source"] = (
            country_source.loc[province_mask]
        )

        country_output = chunk.loc[
            country_mask,
            base_columns,
        ].copy()

        country_output["centroid_target_level"] = "country"
        country_output["country_identifier_source"] = (
            country_source.loc[country_mask]
        )

        unresolved_output = chunk.loc[
            unresolved_province_mask,
            base_columns,
        ].copy()

        unresolved_output["centroid_target_level"] = "unresolved"
        unresolved_output["country_identifier_source"] = (
            "missing_country_and_country_iso"
        )

        province_output.to_csv(
            OUTPUT_PROVINCE,
            mode="a",
            header=False,
            index=False,
        )

        country_output.to_csv(
            OUTPUT_COUNTRY,
            mode="a",
            header=False,
            index=False,
        )

        unresolved_output.to_csv(
            OUTPUT_UNRESOLVED_PROVINCE,
            mode="a",
            header=False,
            index=False,
        )

        totals["total_records"] += chunk_size
        totals["valid_coordinate_records"] += int(has_coord.sum())
        totals["missing_coordinate_records"] += int(missing_coord.sum())
        totals["province_centroid_candidates"] += int(
            province_mask.sum()
        )
        totals["country_centroid_candidates"] += int(
            country_mask.sum()
        )
        totals["province_without_country_or_iso"] += int(
            unresolved_province_mask.sum()
        )
        totals["no_province_country_or_iso"] += int(
            no_admin_mask.sum()
        )

        processed_rows += chunk_size

        print(
            f"Chunk {chunk_number}: {chunk_size:,} rows | "
            f"valid coordinates {has_coord.sum():,} | "
            f"province candidates {province_mask.sum():,} | "
            f"country candidates {country_mask.sum():,} | "
            f"unresolved "
            f"{unresolved_province_mask.sum() + no_admin_mask.sum():,}"
        )

    classified_missing = (
        totals["province_centroid_candidates"]
        + totals["country_centroid_candidates"]
        + totals["province_without_country_or_iso"]
        + totals["no_province_country_or_iso"]
    )

    if classified_missing != totals["missing_coordinate_records"]:
        raise RuntimeError(
            "Classification totals do not match the number of "
            "missing-coordinate records."
        )

    unique_province_count = create_unique_queries(
        OUTPUT_PROVINCE,
        OUTPUT_UNIQUE_PROVINCE,
        [
            "country",
            "country_iso",
            "province_state",
            "country_identifier_source",
        ],
    )

    unique_country_count = create_unique_queries(
        OUTPUT_COUNTRY,
        OUTPUT_UNIQUE_COUNTRY,
        [
            "country",
            "country_iso",
            "country_identifier_source",
        ],
    )

    summary = pd.DataFrame(
        {
            "category": list(totals.keys()) + [
                "unique_province_queries",
                "unique_country_queries",
            ],
            "record_count": list(totals.values()) + [
                unique_province_count,
                unique_country_count,
            ],
        }
    )

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print()
    print("=" * 80)
    print("FINAL SCREENING SUMMARY")
    print("=" * 80)
    print(f"Total records:                       {totals['total_records']:,}")
    print(
        f"Already have valid coordinates:      "
        f"{totals['valid_coordinate_records']:,}"
    )
    print(
        f"Still missing coordinates:           "
        f"{totals['missing_coordinate_records']:,}"
    )
    print(
        f"Province centroid candidates:        "
        f"{totals['province_centroid_candidates']:,}"
    )
    print(
        f"Country centroid candidates:         "
        f"{totals['country_centroid_candidates']:,}"
    )
    print(
        f"Province without country/ISO:        "
        f"{totals['province_without_country_or_iso']:,}"
    )
    print(
        f"No province/country/ISO:             "
        f"{totals['no_province_country_or_iso']:,}"
    )
    print(f"Unique province queries:             {unique_province_count:,}")
    print(f"Unique country queries:              {unique_country_count:,}")
    print()
    print("Classification check passed.")
    print()
    print("Saved:")
    print(f"  {OUTPUT_PROVINCE}")
    print(f"  {OUTPUT_COUNTRY}")
    print(f"  {OUTPUT_UNRESOLVED_PROVINCE}")
    print(f"  {OUTPUT_UNIQUE_PROVINCE}")
    print(f"  {OUTPUT_UNIQUE_COUNTRY}")
    print(f"  {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
