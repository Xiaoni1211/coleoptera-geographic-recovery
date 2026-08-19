#!/usr/bin/env python3

import argparse
from collections import Counter

import pandas as pd


MISSING_TEXT = {
    "",
    "na",
    "nan",
    "none",
    "null",
    "<na>",
}


def nonempty(series):
    """Return True for values that are not missing or blank."""
    text = series.astype("string").str.strip()
    return series.notna() & ~text.str.lower().isin(MISSING_TEXT)


def find_related_columns(columns, keywords):
    found = []

    for col in columns:
        lower = col.lower()

        if any(keyword in lower for keyword in keywords):
            found.append(col)

    return found


def add_samples(container, series, maximum=15):
    """Collect a small number of non-missing example values."""
    if len(container) >= maximum:
        return

    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        ~values.str.lower().isin(MISSING_TEXT)
    ]

    for value in values:
        if value not in container:
            container.append(value)

        if len(container) >= maximum:
            break


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect columns needed for PTP hidden-truth realm validation."
        )
    )

    parser.add_argument(
        "input_csv",
        help="Complete CSV file containing original coordinates and PTP assignments.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Rows per chunk. Default: 100000",
    )

    args = parser.parse_args()

    # Read only the header first.
    header = pd.read_csv(
        args.input_csv,
        nrows=0,
        low_memory=False,
    )

    columns = header.columns.tolist()

    print("=" * 90)
    print("INPUT FILE")
    print("=" * 90)
    print(args.input_csv)

    print("\n" + "=" * 90)
    print("ALL COLUMNS")
    print("=" * 90)

    for i, col in enumerate(columns, start=1):
        print(f"{i:3d}. {col}")

    coordinate_related = find_related_columns(
        columns,
        [
            "coord",
            "latitude",
            "longitude",
            "lat",
            "lon",
            "geocode",
            "centroid",
        ],
    )

    ptp_related = find_related_columns(
        columns,
        ["ptp"],
    )

    realm_related = find_related_columns(
        columns,
        ["realm"],
    )

    identifier_related = find_related_columns(
        columns,
        [
            "record_id",
            "sampleid",
            "specimen",
            "btseq",
            "db_id",
            "species",
        ],
    )

    print("\n" + "=" * 90)
    print("POSSIBLE COORDINATE COLUMNS")
    print("=" * 90)
    print(coordinate_related)

    print("\n" + "=" * 90)
    print("POSSIBLE PTP COLUMNS")
    print("=" * 90)
    print(ptp_related)

    print("\n" + "=" * 90)
    print("POSSIBLE REALM COLUMNS")
    print("=" * 90)
    print(realm_related)

    print("\n" + "=" * 90)
    print("POSSIBLE IDENTIFIER/SPECIES COLUMNS")
    print("=" * 90)
    print(identifier_related)

    # Columns currently expected from your project.
    expected_columns = [
        "coord",
        "realm",
        "ptp_species",
        "species",
        "record_id",
        "BTseq_id",
        "db_id",
        "final_latitude",
        "final_longitude",
        "final_geocode_source",
        "coordinate_source",
        "final_coordinate_source",
        "final_coordinate_valid",
    ]

    selected_columns = [
        col for col in expected_columns
        if col in columns
    ]

    # Include other detected columns in case the actual names differ.
    for col in (
        coordinate_related
        + ptp_related
        + realm_related
        + identifier_related
    ):
        if col not in selected_columns:
            selected_columns.append(col)

    print("\n" + "=" * 90)
    print("COLUMNS THAT WILL BE INSPECTED")
    print("=" * 90)
    print(selected_columns)

    if not selected_columns:
        raise ValueError(
            "No relevant columns were detected."
        )

    total_rows = 0
    nonmissing_counts = Counter()
    realm_counts = Counter()

    coordinate_samples = []
    realm_samples = []
    ptp_samples = []

    coord_col = "coord" if "coord" in columns else None
    realm_col = "realm" if "realm" in columns else None
    ptp_col = "ptp_species" if "ptp_species" in columns else None

    candidate_rows = 0
    candidate_with_realm = 0
    candidate_without_realm = 0

    print("\n" + "=" * 90)
    print("SCANNING FILE IN CHUNKS")
    print("=" * 90)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            args.input_csv,
            usecols=selected_columns,
            chunksize=args.chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        for col in selected_columns:
            nonmissing_counts[col] += int(
                nonempty(chunk[col]).sum()
            )

        if coord_col:
            add_samples(
                coordinate_samples,
                chunk[coord_col],
            )

        if realm_col:
            add_samples(
                realm_samples,
                chunk[realm_col],
            )

            valid_realm = nonempty(chunk[realm_col])

            for value, count in (
                chunk.loc[valid_realm, realm_col]
                .astype(str)
                .str.strip()
                .value_counts()
                .items()
            ):
                realm_counts[value] += int(count)

        if ptp_col:
            add_samples(
                ptp_samples,
                chunk[ptp_col],
            )

        # Initial candidate count based on non-empty raw coord and PTP.
        # This is only a screening count; coordinate validity will be
        # checked properly in the final validation script.
        if coord_col and ptp_col:
            has_coord = nonempty(chunk[coord_col])
            has_ptp = nonempty(chunk[ptp_col])

            candidate_mask = has_coord & has_ptp
            candidate_rows += int(candidate_mask.sum())

            if realm_col:
                has_realm = nonempty(chunk[realm_col])

                candidate_with_realm += int(
                    (candidate_mask & has_realm).sum()
                )

                candidate_without_realm += int(
                    (candidate_mask & ~has_realm).sum()
                )

        print(
            f"Chunk {chunk_number}: "
            f"{len(chunk):,} rows | "
            f"cumulative rows {total_rows:,}"
        )

    print("\n" + "=" * 90)
    print("NON-MISSING COUNTS")
    print("=" * 90)

    print(f"Total rows: {total_rows:,}")

    for col in selected_columns:
        count = nonmissing_counts[col]
        proportion = count / total_rows if total_rows else 0

        print(
            f"{col:35s} "
            f"{count:12,d} "
            f"({proportion:7.2%})"
        )

    print("\n" + "=" * 90)
    print("INITIAL PTP VALIDATION CANDIDATE SCREEN")
    print("=" * 90)

    if coord_col and ptp_col:
        print(
            "Rows with non-empty coord and ptp_species: "
            f"{candidate_rows:,}"
        )

        if realm_col:
            print(
                "Candidate rows with realm:             "
                f"{candidate_with_realm:,}"
            )

            print(
                "Candidate rows without realm:          "
                f"{candidate_without_realm:,}"
            )

            coverage = (
                candidate_with_realm / candidate_rows
                if candidate_rows
                else 0
            )

            print(
                "Existing realm coverage:               "
                f"{coverage:.2%}"
            )
    else:
        print(
            "Could not calculate candidate coverage because "
            "'coord' or 'ptp_species' was not found."
        )

    print("\n" + "=" * 90)
    print("REALM VALUE COUNTS")
    print("=" * 90)

    if realm_counts:
        for realm, count in realm_counts.most_common():
            print(f"{realm:35s} {count:12,d}")
    else:
        print("No non-missing realm values found.")

    print("\n" + "=" * 90)
    print("EXAMPLE RAW COORD VALUES")
    print("=" * 90)

    for value in coordinate_samples:
        print(repr(value))

    print("\n" + "=" * 90)
    print("EXAMPLE REALM VALUES")
    print("=" * 90)

    for value in realm_samples:
        print(repr(value))

    print("\n" + "=" * 90)
    print("EXAMPLE PTP VALUES")
    print("=" * 90)

    for value in ptp_samples:
        print(repr(value))


if __name__ == "__main__":
    main()
