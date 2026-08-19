#!/usr/bin/env python3

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


MISSING_TEXT = {"", "na", "nan", "none", "null", "<na>"}


def clean_text(series):
    text = series.astype("string").str.strip()
    return text.mask(text.str.lower().isin(MISSING_TEXT))


def find_column(columns, candidates):
    lookup = {str(col).lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(columns)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare original-coordinate PTP clusters for "
            "hidden-truth realm validation."
        )
    )

    parser.add_argument(
        "input_csv",
        help="Complete dataset containing original coordinates and ptp_species.",
    )

    parser.add_argument(
        "--realm-file",
        default="realm_data/resolve_ecoregions_2017/Ecoregions2017.shp",
        help="RESOLVE Ecoregions2017 shapefile.",
    )

    parser.add_argument(
        "--output-dir",
        default="ptp_hidden_truth_validation",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=4,
        help=(
            "Decimal places used to define independent locations. "
            "Default 4 is approximately 11 metres."
        ),
    )

    parser.add_argument(
        "--min-remaining-locations",
        type=int,
        default=3,
        help=(
            "Minimum independent reference locations remaining "
            "after one location is hidden."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = [
        "BTseq_id",
        "species",
        "ptp_species",
        "original_latitude",
        "original_longitude",
        "realm",
    ]

    header = pd.read_csv(
        args.input_csv,
        nrows=0,
        low_memory=False,
    )

    missing_columns = [
        col for col in required_columns
        if col not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    print("=" * 100)
    print("PASS 1: EXTRACT ORIGINAL-COORDINATE PTP EVIDENCE")
    print("=" * 100)
    print(f"Input:  {args.input_csv}")
    print(f"Realm:  {args.realm_file}")
    print(f"Output: {output_dir}")
    print(
        "Independent-location precision: "
        f"{args.coordinate_decimals} decimal places"
    )

    total_rows = 0
    rows_with_ptp = 0
    valid_candidate_records = 0
    invalid_coordinate_records = 0

    location_parts = []
    species_parts = []

    reader = pd.read_csv(
        args.input_csv,
        usecols=required_columns,
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        chunk["ptp_species"] = clean_text(
            chunk["ptp_species"]
        )

        chunk["species"] = clean_text(
            chunk["species"]
        )

        chunk["original_latitude"] = pd.to_numeric(
            chunk["original_latitude"],
            errors="coerce",
        )

        chunk["original_longitude"] = pd.to_numeric(
            chunk["original_longitude"],
            errors="coerce",
        )

        has_ptp = chunk["ptp_species"].notna()

        valid_coordinate = (
            chunk["original_latitude"].between(-90, 90)
            & chunk["original_longitude"].between(-180, 180)
        )

        rows_with_ptp += int(has_ptp.sum())

        invalid_coordinate_records += int(
            (has_ptp & ~valid_coordinate).sum()
        )

        candidate_mask = has_ptp & valid_coordinate

        candidate = chunk.loc[
            candidate_mask,
            [
                "BTseq_id",
                "species",
                "ptp_species",
                "original_latitude",
                "original_longitude",
            ],
        ].copy()

        valid_candidate_records += len(candidate)

        candidate["location_latitude"] = (
            candidate["original_latitude"]
            .round(args.coordinate_decimals)
        )

        candidate["location_longitude"] = (
            candidate["original_longitude"]
            .round(args.coordinate_decimals)
        )

        # Count records at each independent location within each cluster.
        location_part = (
            candidate.groupby(
                [
                    "ptp_species",
                    "location_latitude",
                    "location_longitude",
                ],
                as_index=False,
            )
            .agg(
                records_at_location_n=(
                    "BTseq_id",
                    "size",
                )
            )
        )

        location_parts.append(location_part)

        species_part = (
            candidate.loc[
                candidate["species"].notna(),
                ["ptp_species", "species"],
            ]
            .drop_duplicates()
        )

        species_parts.append(species_part)

        print(
            f"Chunk {chunk_number}: "
            f"{len(chunk):,} rows | "
            f"valid original-coordinate PTP records "
            f"{len(candidate):,}"
        )

    cluster_locations = pd.concat(
        location_parts,
        ignore_index=True,
    )

    # The same cluster-location may occur in more than one input chunk.
    cluster_locations = (
        cluster_locations.groupby(
            [
                "ptp_species",
                "location_latitude",
                "location_longitude",
            ],
            as_index=False,
        )
        .agg(
            records_at_location_n=(
                "records_at_location_n",
                "sum",
            )
        )
    )

    species_pairs = pd.concat(
        species_parts,
        ignore_index=True,
    ).drop_duplicates()

    species_summary = (
        species_pairs.groupby(
            "ptp_species",
            as_index=False,
        )
        .agg(
            species_names_n=("species", "nunique"),
            example_species=("species", "first"),
        )
    )

    print("\n" + "=" * 100)
    print("PASS 2: BUILD UNIQUE COORDINATE LOOKUP")
    print("=" * 100)

    unique_locations = (
        cluster_locations[
            [
                "location_latitude",
                "location_longitude",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    unique_locations["_location_id"] = np.arange(
        len(unique_locations),
        dtype=np.int64,
    )

    print(
        f"Unique coordinate locations requiring realm overlay: "
        f"{len(unique_locations):,}"
    )

    print("\n" + "=" * 100)
    print("PASS 3: LOAD RESOLVE REALM POLYGONS")
    print("=" * 100)

    realms = gpd.read_file(args.realm_file)

    print(f"Polygons loaded: {len(realms):,}")
    print(f"Original CRS:    {realms.crs}")
    print(f"Columns:         {realms.columns.tolist()}")

    realm_col = find_column(
        realms.columns,
        ["REALM", "realm", "REALM_NAME"],
    )

    realms = realms[
        [realm_col, "geometry"]
    ].copy()

    realms = realms.rename(
        columns={realm_col: "assigned_realm"}
    )

    realms["assigned_realm"] = clean_text(
        realms["assigned_realm"]
    )

    # Correct the non-standard label seen in the original data.
    oceania_mask = (
        realms["assigned_realm"]
        .str.startswith(
            "Oceania_simplify",
            na=False,
        )
    )

    realms.loc[
        oceania_mask,
        "assigned_realm",
    ] = "Oceania"

    realms = realms[
        realms.geometry.notna()
        & ~realms.geometry.is_empty
    ].copy()

    if realms.crs is None:
        raise ValueError(
            "Realm shapefile has no CRS."
        )

    realms = realms.to_crs("EPSG:4326")

    print(f"Realm column: {realm_col}")
    print(
        "Realm values: "
        f"{sorted(realms['assigned_realm'].dropna().unique())}"
    )

    print("\n" + "=" * 100)
    print("PASS 4: ASSIGN REALMS TO UNIQUE ORIGINAL LOCATIONS")
    print("=" * 100)

    points = gpd.GeoDataFrame(
        unique_locations.copy(),
        geometry=gpd.points_from_xy(
            unique_locations["location_longitude"],
            unique_locations["location_latitude"],
        ),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points,
        realms,
        how="left",
        predicate="intersects",
    )

    match_summary = (
        joined.groupby(
            "_location_id",
            as_index=False,
        )
        .agg(
            realm_polygon_matches_n=(
                "assigned_realm",
                "count",
            ),
            distinct_realms_matched_n=(
                "assigned_realm",
                "nunique",
            ),
        )
    )

    realm_values = (
        joined.loc[
            joined["assigned_realm"].notna(),
            ["_location_id", "assigned_realm"],
        ]
        .drop_duplicates()
        .sort_values(
            ["_location_id", "assigned_realm"]
        )
    )

    single_realm = (
        realm_values.groupby(
            "_location_id",
            as_index=False,
        )
        .agg(
            validation_true_realm=(
                "assigned_realm",
                "first",
            )
        )
    )

    location_lookup = (
        unique_locations.merge(
            match_summary,
            on="_location_id",
            how="left",
        )
        .merge(
            single_realm,
            on="_location_id",
            how="left",
        )
    )

    location_lookup[
        "realm_assignment_status"
    ] = "assigned"

    no_match = (
        location_lookup[
            "distinct_realms_matched_n"
        ].fillna(0).eq(0)
    )

    same_realm_multiple = (
        location_lookup[
            "realm_polygon_matches_n"
        ].fillna(0).gt(1)
        & location_lookup[
            "distinct_realms_matched_n"
        ].fillna(0).eq(1)
    )

    ambiguous = (
        location_lookup[
            "distinct_realms_matched_n"
        ].fillna(0).gt(1)
    )

    location_lookup.loc[
        no_match,
        "realm_assignment_status",
    ] = "no_realm_polygon_match"

    location_lookup.loc[
        same_realm_multiple,
        "realm_assignment_status",
    ] = "assigned_multiple_polygons_same_realm"

    location_lookup.loc[
        ambiguous,
        "realm_assignment_status",
    ] = "ambiguous_realm_boundary"

    # Do not use ambiguous boundary points as truth.
    location_lookup.loc[
        ambiguous,
        "validation_true_realm",
    ] = pd.NA

    location_lookup[
        "realm_polygon_matches_n"
    ] = (
        location_lookup[
            "realm_polygon_matches_n"
        ]
        .fillna(0)
        .astype(int)
    )

    location_lookup[
        "distinct_realms_matched_n"
    ] = (
        location_lookup[
            "distinct_realms_matched_n"
        ]
        .fillna(0)
        .astype(int)
    )

    location_lookup.to_csv(
        output_dir
        / "ptp_original_coordinate_realm_lookup.csv",
        index=False,
    )

    print(
        location_lookup[
            "realm_assignment_status"
        ].value_counts(dropna=False)
    )

    print("\n" + "=" * 100)
    print("PASS 5: BUILD CLUSTER-LOCATION EVIDENCE")
    print("=" * 100)

    evidence = cluster_locations.merge(
        location_lookup[
            [
                "location_latitude",
                "location_longitude",
                "validation_true_realm",
                "realm_assignment_status",
                "realm_polygon_matches_n",
                "distinct_realms_matched_n",
            ]
        ],
        on=[
            "location_latitude",
            "location_longitude",
        ],
        how="left",
        validate="many_to_one",
    )

    evidence.to_csv(
        output_dir
        / "ptp_cluster_location_realm_evidence.csv",
        index=False,
    )

    assigned_evidence = evidence.loc[
        evidence["validation_true_realm"].notna()
    ].copy()

    realm_composition = (
        assigned_evidence.groupby(
            [
                "ptp_species",
                "validation_true_realm",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "independent_locations_n"
            }
        )
    )

    realm_totals = (
        realm_composition.groupby(
            "ptp_species"
        )["independent_locations_n"]
        .transform("sum")
    )

    realm_composition[
        "realm_location_proportion"
    ] = (
        realm_composition[
            "independent_locations_n"
        ]
        / realm_totals
    )

    realm_composition.to_csv(
        output_dir
        / "ptp_cluster_realm_composition.csv",
        index=False,
    )

    cluster_summary = (
        evidence.groupby(
            "ptp_species",
            as_index=False,
        )
        .agg(
            original_coordinate_records_n=(
                "records_at_location_n",
                "sum",
            ),
            independent_locations_n=(
                "ptp_species",
                "size",
            ),
            assigned_realm_locations_n=(
                "validation_true_realm",
                lambda x: int(x.notna().sum()),
            ),
            ambiguous_locations_n=(
                "realm_assignment_status",
                lambda x: int(
                    (
                        x
                        == "ambiguous_realm_boundary"
                    ).sum()
                ),
            ),
            unmatched_locations_n=(
                "realm_assignment_status",
                lambda x: int(
                    (
                        x
                        == "no_realm_polygon_match"
                    ).sum()
                ),
            ),
        )
    )

    if not realm_composition.empty:
        ranked = realm_composition.sort_values(
            [
                "ptp_species",
                "independent_locations_n",
                "validation_true_realm",
            ],
            ascending=[True, False, True],
        )

        dominant = (
            ranked.drop_duplicates(
                "ptp_species"
            )
            [
                [
                    "ptp_species",
                    "validation_true_realm",
                    "independent_locations_n",
                    "realm_location_proportion",
                ]
            ]
            .rename(
                columns={
                    "validation_true_realm":
                        "dominant_realm",
                    "independent_locations_n":
                        "dominant_realm_locations_n",
                    "realm_location_proportion":
                        "dominant_realm_proportion",
                }
            )
        )

        realm_numbers = (
            realm_composition.groupby(
                "ptp_species",
                as_index=False,
            )
            .agg(
                realms_detected_n=(
                    "validation_true_realm",
                    "nunique",
                )
            )
        )

        cluster_summary = (
            cluster_summary.merge(
                dominant,
                on="ptp_species",
                how="left",
            )
            .merge(
                realm_numbers,
                on="ptp_species",
                how="left",
            )
        )

    cluster_summary = cluster_summary.merge(
        species_summary,
        on="ptp_species",
        how="left",
    )

    required_before_hiding = (
        args.min_remaining_locations + 1
    )

    cluster_summary[
        "eligible_for_hidden_validation"
    ] = (
        cluster_summary[
            "assigned_realm_locations_n"
        ]
        >= required_before_hiding
    )

    cluster_summary[
        "eligible_hidden_locations_n"
    ] = np.where(
        cluster_summary[
            "eligible_for_hidden_validation"
        ],
        cluster_summary[
            "assigned_realm_locations_n"
        ],
        0,
    )

    cluster_summary.to_csv(
        output_dir
        / "ptp_hidden_truth_cluster_eligibility.csv",
        index=False,
    )

    eligible_clusters = int(
        cluster_summary[
            "eligible_for_hidden_validation"
        ].sum()
    )

    eligible_locations = int(
        cluster_summary[
            "eligible_hidden_locations_n"
        ].sum()
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "input_rows",
                "rows_with_ptp_species",
                "valid_original_coordinate_ptp_records",
                "invalid_or_missing_original_coordinate_ptp_records",
                "unique_original_coordinate_locations",
                "ptp_cluster_location_combinations",
                "ptp_clusters_with_original_coordinate_evidence",
                "minimum_locations_required_before_hiding",
                "eligible_ptp_clusters",
                "eligible_hidden_locations",
            ],
            "value": [
                total_rows,
                rows_with_ptp,
                valid_candidate_records,
                invalid_coordinate_records,
                len(unique_locations),
                len(evidence),
                cluster_summary[
                    "ptp_species"
                ].nunique(),
                required_before_hiding,
                eligible_clusters,
                eligible_locations,
            ],
        }
    )

    summary.to_csv(
        output_dir
        / "ptp_hidden_truth_pool_summary.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nSaved files:")

    for path in sorted(
        output_dir.glob("*.csv")
    ):
        print(path)

    print(
        "\nPreparation completed. "
        "No coordinates were hidden and no missing records "
        "were predicted in this step."
    )


if __name__ == "__main__":
    main()
