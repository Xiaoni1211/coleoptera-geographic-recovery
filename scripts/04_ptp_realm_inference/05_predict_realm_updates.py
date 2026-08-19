#!/usr/bin/env python3

"""
Calculate PTP realm predictions and confidence for records that:

1. Still have no effective realm after applying the direct-coordinate
   realm update table;
2. Have no valid final coordinate;
3. Have a non-missing ptp_species;
4. Belong to a PTP cluster represented in the original-coordinate
   realm-evidence table.

Important rules
---------------
- PTP evidence is based only on independent original-coordinate locations.
- Each independent location has equal weight.
- records_at_location_n is not used as prediction weight.
- Direct-coordinate realm updates are applied virtually, in chunks.
- The main 1.83-million-row file is not rewritten.
- input_row_number is the physical zero-based row number in the master CSV.
- BTseq_id is checked before any direct realm update is applied.

Confidence rules copied from the completed hidden-truth validation:
- High:
    support locations >= 5 and dominant realm proportion >= 0.90
- Medium:
    support locations >= 3 and dominant realm proportion >= 0.70,
    but not High
- Low:
    support locations >= 3 and dominant realm proportion < 0.70
- Insufficient:
    support locations < 3 or dominant realm is tied

High and Medium are eligible for automatic realm assignment.
Low and Insufficient are retained for reporting but are not assigned.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_MASTER = Path(
    "centroid_results/new_all_seq_with_final_centroids.csv"
)

DEFAULT_EVIDENCE = Path(
    "ptp_hidden_truth_validation/"
    "ptp_cluster_location_realm_evidence.csv"
)

DEFAULT_DIRECT_UPDATES = Path(
    "direct_coordinate_realm_assignment/"
    "direct_coordinate_realm_update_table.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "ptp_realm_prediction"
)

VALID_REALM_STATUSES = {
    "assigned",
    "assigned_multiple_polygons_same_realm",
}

MIN_SUPPORT = 3
HIGH_MIN_LOCATIONS = 5
HIGH_PURITY = 0.90
MEDIUM_PURITY = 0.70

EXPECTED_MASTER_ROWS = 1_827_573
EXPECTED_INFERENCE_TARGETS = 831_921
EXPECTED_POTENTIAL_PTP = 166_462


# =============================================================================
# HELPERS
# =============================================================================

def normalise_text(series: pd.Series) -> pd.Series:
    """Return stripped pandas string values with common missing tokens removed."""

    result = series.astype("string").str.strip()

    missing_tokens = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "<na>",
    }

    lower = result.str.lower()
    return result.mask(lower.isin(missing_tokens), pd.NA)


def parse_boolean(series: pd.Series) -> pd.Series:
    """Convert common Boolean representations to pandas nullable Boolean."""

    text = series.astype("string").str.strip().str.lower()

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "t": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "f": False,
    }

    return text.map(mapping).astype("boolean")


def actual_valid_coordinate(
    latitude: pd.Series,
    longitude: pd.Series,
) -> pd.Series:
    """
    Determine validity from the actual final coordinate values.

    A valid coordinate must:
    - contain numeric latitude and longitude;
    - fall within geographic bounds;
    - not be exactly (0, 0).
    """

    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")

    valid = (
        lat.notna()
        & lon.notna()
        & lat.between(-90, 90, inclusive="both")
        & lon.between(-180, 180, inclusive="both")
        & ~((lat == 0) & (lon == 0))
    )

    return valid.astype(bool)


def append_csv(
    dataframe: pd.DataFrame,
    path: Path,
    write_header: bool,
) -> None:
    """Append a dataframe to a CSV, writing the header only once."""

    if dataframe.empty:
        return

    dataframe.to_csv(
        path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def check_required_columns(
    path: Path,
    required: set[str],
    label: str,
) -> list[str]:
    """Check that a CSV exists and contains all required columns."""

    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(required - set(columns))

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {columns}"
        )

    return columns


# =============================================================================
# PTP CLUSTER PREDICTIONS
# =============================================================================

def build_ptp_cluster_predictions(
    evidence_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Build one realm prediction per PTP cluster.

    The evidence file may contain assigned and unassigned independent
    original-coordinate locations. All coordinate locations are counted
    for overlay-coverage reporting, but only locations with a valid
    terrestrial realm assignment vote in the realm prediction.
    """

    print("=" * 100)
    print("BUILD PTP CLUSTER REALM PREDICTIONS")
    print("=" * 100)
    print(f"Evidence: {evidence_path}")

    required = {
        "ptp_species",
        "location_latitude",
        "location_longitude",
        "validation_true_realm",
        "realm_assignment_status",
    }

    check_required_columns(
        evidence_path,
        required,
        "PTP evidence file",
    )

    evidence = pd.read_csv(
        evidence_path,
        usecols=[
            "ptp_species",
            "location_latitude",
            "location_longitude",
            "validation_true_realm",
            "realm_assignment_status",
        ],
        dtype={
            "ptp_species": "string",
            "validation_true_realm": "string",
            "realm_assignment_status": "string",
        },
        low_memory=False,
    )

    raw_rows_n = len(evidence)

    evidence["ptp_species"] = normalise_text(
        evidence["ptp_species"]
    )
    evidence["validation_true_realm"] = normalise_text(
        evidence["validation_true_realm"]
    )
    evidence["realm_assignment_status"] = normalise_text(
        evidence["realm_assignment_status"]
    ).str.lower()

    evidence["location_latitude"] = pd.to_numeric(
        evidence["location_latitude"],
        errors="coerce",
    )
    evidence["location_longitude"] = pd.to_numeric(
        evidence["location_longitude"],
        errors="coerce",
    )

    valid_coordinate = (
        evidence["location_latitude"].notna()
        & evidence["location_longitude"].notna()
        & evidence["location_latitude"].between(-90, 90)
        & evidence["location_longitude"].between(-180, 180)
        & ~(
            (evidence["location_latitude"] == 0)
            & (evidence["location_longitude"] == 0)
        )
    )

    evidence = evidence.loc[
        evidence["ptp_species"].notna()
        & valid_coordinate
    ].copy()

    location_keys = [
        "ptp_species",
        "location_latitude",
        "location_longitude",
    ]

    # All independent original-coordinate locations, including locations
    # that did not receive a valid terrestrial realm.
    all_locations = evidence[
        location_keys
    ].drop_duplicates()

    all_location_counts = (
        all_locations
        .groupby("ptp_species", observed=True)
        .size()
        .rename("original_coordinate_locations_n")
        .reset_index()
    )

    # Only valid realm assignments are allowed to vote.
    valid_realm_evidence = evidence.loc[
        evidence["realm_assignment_status"].isin(
            VALID_REALM_STATUSES
        )
        & evidence["validation_true_realm"].notna(),
        location_keys + ["validation_true_realm"],
    ].drop_duplicates()

    # Detect the unlikely case where exactly the same cluster/location
    # has conflicting assigned realms. Such locations are excluded rather
    # than arbitrarily choosing one realm.
    realms_per_location = (
        valid_realm_evidence
        .groupby(location_keys, observed=True)[
            "validation_true_realm"
        ]
        .nunique()
        .rename("distinct_realms_at_location_n")
        .reset_index()
    )

    conflicting_locations = realms_per_location.loc[
        realms_per_location[
            "distinct_realms_at_location_n"
        ] > 1,
        location_keys,
    ]

    conflicting_locations_n = len(conflicting_locations)

    valid_realm_evidence = valid_realm_evidence.merge(
        realms_per_location,
        on=location_keys,
        how="left",
        validate="many_to_one",
    )

    valid_realm_evidence = valid_realm_evidence.loc[
        valid_realm_evidence[
            "distinct_realms_at_location_n"
        ] == 1
    ].drop(
        columns="distinct_realms_at_location_n"
    )

    valid_realm_evidence = (
        valid_realm_evidence
        .drop_duplicates(location_keys)
    )

    # Count independent locations in each realm.
    composition = (
        valid_realm_evidence
        .groupby(
            ["ptp_species", "validation_true_realm"],
            observed=True,
        )
        .size()
        .rename("realm_locations_n")
        .reset_index()
    )

    assigned_counts = (
        composition
        .groupby("ptp_species", observed=True)[
            "realm_locations_n"
        ]
        .sum()
        .rename("assigned_realm_locations_n")
        .reset_index()
    )

    realms_detected = (
        composition
        .groupby("ptp_species", observed=True)
        .size()
        .rename("realms_detected_n")
        .reset_index()
    )

    dominant_counts = (
        composition
        .groupby("ptp_species", observed=True)[
            "realm_locations_n"
        ]
        .max()
        .rename("dominant_realm_locations_n")
        .reset_index()
    )

    composition_with_max = composition.merge(
        dominant_counts,
        on="ptp_species",
        how="left",
        validate="many_to_one",
    )

    dominant_rows = composition_with_max.loc[
        composition_with_max["realm_locations_n"]
        == composition_with_max[
            "dominant_realm_locations_n"
        ]
    ].copy()

    tied_counts = (
        dominant_rows
        .groupby("ptp_species", observed=True)
        .size()
        .rename("dominant_tied_realms_n")
        .reset_index()
    )

    # Sorting makes the candidate deterministic, although tied predictions
    # will later be classified as Insufficient and not assigned.
    dominant_candidates = (
        dominant_rows
        .sort_values(
            ["ptp_species", "validation_true_realm"]
        )
        .drop_duplicates(
            subset=["ptp_species"],
            keep="first",
        )
        .rename(
            columns={
                "validation_true_realm":
                    "ptp_predicted_realm_candidate"
            }
        )[
            [
                "ptp_species",
                "ptp_predicted_realm_candidate",
            ]
        ]
    )

    clusters = all_location_counts.copy()

    for table in [
        assigned_counts,
        realms_detected,
        dominant_counts,
        tied_counts,
        dominant_candidates,
    ]:
        clusters = clusters.merge(
            table,
            on="ptp_species",
            how="left",
            validate="one_to_one",
        )

    count_columns = [
        "assigned_realm_locations_n",
        "realms_detected_n",
        "dominant_realm_locations_n",
        "dominant_tied_realms_n",
    ]

    for column in count_columns:
        clusters[column] = (
            clusters[column]
            .fillna(0)
            .astype("int64")
        )

    clusters["dominant_realm_proportion"] = np.where(
        clusters["assigned_realm_locations_n"] > 0,
        (
            clusters["dominant_realm_locations_n"]
            / clusters["assigned_realm_locations_n"]
        ),
        np.nan,
    )

    clusters["realm_overlay_coverage"] = np.where(
        clusters["original_coordinate_locations_n"] > 0,
        (
            clusters["assigned_realm_locations_n"]
            / clusters["original_coordinate_locations_n"]
        ),
        np.nan,
    )

    sufficient_support = (
        clusters["assigned_realm_locations_n"]
        >= MIN_SUPPORT
    )

    unique_dominant = (
        clusters["dominant_tied_realms_n"] == 1
    )

    high = (
        sufficient_support
        & unique_dominant
        & (
            clusters["assigned_realm_locations_n"]
            >= HIGH_MIN_LOCATIONS
        )
        & (
            clusters["dominant_realm_proportion"]
            >= HIGH_PURITY
        )
    )

    medium = (
        sufficient_support
        & unique_dominant
        & ~high
        & (
            clusters["dominant_realm_proportion"]
            >= MEDIUM_PURITY
        )
    )

    low = (
        sufficient_support
        & unique_dominant
        & ~high
        & ~medium
    )

    clusters["ptp_realm_confidence"] = "Insufficient"
    clusters.loc[low, "ptp_realm_confidence"] = "Low"
    clusters.loc[medium, "ptp_realm_confidence"] = "Medium"
    clusters.loc[high, "ptp_realm_confidence"] = "High"

    # A formal prediction is retained for High, Medium and Low.
    # Insufficient clusters do not receive a predicted realm.
    formal_prediction = sufficient_support & unique_dominant

    clusters["ptp_predicted_realm"] = (
        clusters["ptp_predicted_realm_candidate"]
        .where(formal_prediction, pd.NA)
        .astype("string")
    )

    clusters["ptp_auto_assignment_eligible"] = (
        clusters["ptp_realm_confidence"].isin(
            ["High", "Medium"]
        )
        & clusters["ptp_predicted_realm"].notna()
    )

    clusters = clusters[
        [
            "ptp_species",
            "ptp_predicted_realm",
            "ptp_realm_confidence",
            "original_coordinate_locations_n",
            "assigned_realm_locations_n",
            "dominant_realm_locations_n",
            "dominant_realm_proportion",
            "realms_detected_n",
            "dominant_tied_realms_n",
            "realm_overlay_coverage",
            "ptp_auto_assignment_eligible",
        ]
    ].sort_values(
        ["ptp_realm_confidence", "ptp_species"]
    )

    cluster_summary = {
        "raw_evidence_rows_n": int(raw_rows_n),
        "valid_independent_original_locations_n": int(
            len(all_locations)
        ),
        "valid_assigned_realm_locations_n": int(
            len(valid_realm_evidence)
        ),
        "conflicting_locations_excluded_n": int(
            conflicting_locations_n
        ),
        "ptp_clusters_with_original_locations_n": int(
            len(clusters)
        ),
        "ptp_clusters_high_n": int(
            (clusters["ptp_realm_confidence"] == "High").sum()
        ),
        "ptp_clusters_medium_n": int(
            (clusters["ptp_realm_confidence"] == "Medium").sum()
        ),
        "ptp_clusters_low_n": int(
            (clusters["ptp_realm_confidence"] == "Low").sum()
        ),
        "ptp_clusters_insufficient_n": int(
            (
                clusters["ptp_realm_confidence"]
                == "Insufficient"
            ).sum()
        ),
    }

    print(f"Raw evidence rows: {raw_rows_n:,}")
    print(
        "Independent original-coordinate locations: "
        f"{len(all_locations):,}"
    )
    print(
        "Locations with valid assigned realm: "
        f"{len(valid_realm_evidence):,}"
    )
    print(
        "Conflicting locations excluded: "
        f"{conflicting_locations_n:,}"
    )
    print(f"PTP clusters: {len(clusters):,}")
    print("\nCluster confidence:")
    print(
        clusters["ptp_realm_confidence"]
        .value_counts(dropna=False)
        .to_string()
    )

    return clusters, cluster_summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate PTP realm predictions and create a small "
            "record-level realm update table."
        )
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )
    parser.add_argument(
        "--direct-updates",
        type=Path,
        default=DEFAULT_DIRECT_UPDATES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "Finalize output even if the expected counts of "
            "831,921 inference targets and 166,462 Potential PTP "
            "records are not reproduced."
        ),
    )

    args = parser.parse_args()

    required_master = {
        "BTseq_id",
        "db_id",
        "ptp_species",
        "realm",
        "final_latitude",
        "final_longitude",
        "final_coordinate_valid",
    }

    required_direct = {
        "input_row_number",
        "BTseq_id",
        "assigned_realm",
    }

    check_required_columns(
        args.master,
        required_master,
        "Master file",
    )

    check_required_columns(
        args.direct_updates,
        required_direct,
        "Direct realm update table",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_output = (
        args.output_dir
        / "ptp_cluster_realm_predictions.csv"
    )

    all_output = (
        args.output_dir
        / "ptp_record_realm_predictions_all.csv"
    )

    update_output = (
        args.output_dir
        / "ptp_realm_update_table.csv"
    )

    summary_output = (
        args.output_dir
        / "ptp_prediction_summary.csv"
    )

    config_output = (
        args.output_dir
        / "ptp_prediction_config.json"
    )

    # Temporary record-level files are finalized only after the expected
    # target counts and row identifiers have been verified.
    all_temp = all_output.with_suffix(".csv.tmp")
    update_temp = update_output.with_suffix(".csv.tmp")

    print("=" * 100)
    print("PTP REALM PREDICTION")
    print("=" * 100)
    print(f"Master:          {args.master}")
    print(f"Evidence:        {args.evidence}")
    print(f"Direct updates:  {args.direct_updates}")
    print(f"Output:          {args.output_dir}")
    print(f"Chunk size:      {args.chunksize:,}")

    # -------------------------------------------------------------------------
    # Build cluster-level predictions
    # -------------------------------------------------------------------------

    cluster_predictions, cluster_summary = (
        build_ptp_cluster_predictions(args.evidence)
    )

    cluster_predictions.to_csv(
        cluster_output,
        index=False,
    )

    print(f"\nSaved cluster predictions: {cluster_output}")

    cluster_lookup = cluster_predictions.set_index(
        "ptp_species"
    )

    candidate_cluster_names = set(
        cluster_lookup.index.astype(str)
    )

    # -------------------------------------------------------------------------
    # Load the small direct-coordinate update table
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("LOAD DIRECT REALM UPDATE TABLE")
    print("=" * 100)

    direct = pd.read_csv(
        args.direct_updates,
        usecols=[
            "input_row_number",
            "BTseq_id",
            "assigned_realm",
        ],
        dtype={
            "input_row_number": "int64",
            "BTseq_id": "string",
            "assigned_realm": "string",
        },
    )

    direct["BTseq_id"] = normalise_text(
        direct["BTseq_id"]
    )
    direct["assigned_realm"] = normalise_text(
        direct["assigned_realm"]
    )

    if direct["input_row_number"].duplicated().any():
        duplicates = int(
            direct["input_row_number"].duplicated(
                keep=False
            ).sum()
        )
        raise RuntimeError(
            "Direct update table contains duplicate "
            f"input_row_number values: {duplicates:,}"
        )

    if direct["BTseq_id"].isna().any():
        raise RuntimeError(
            "Direct update table contains missing BTseq_id values."
        )

    if direct["assigned_realm"].isna().any():
        raise RuntimeError(
            "Direct update table contains missing assigned_realm values."
        )

    direct = direct.set_index(
        "input_row_number",
        verify_integrity=True,
    ).sort_index()

    print(f"Direct update records: {len(direct):,}")
    print(
        "Unique BTseq_id values: "
        f"{direct['BTseq_id'].nunique():,}"
    )

    # -------------------------------------------------------------------------
    # Scan master file in chunks
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("SCAN MASTER AND APPLY VIRTUAL DIRECT REALM UPDATES")
    print("=" * 100)

    usecols = [
        "BTseq_id",
        "db_id",
        "ptp_species",
        "realm",
        "final_latitude",
        "final_longitude",
        "final_coordinate_valid",
    ]

    dtype = {
        "BTseq_id": "string",
        "db_id": "string",
        "ptp_species": "string",
        "realm": "string",
    }

    total_rows = 0
    inference_targets_n = 0
    potential_ptp_n = 0
    direct_updates_matched_n = 0
    direct_updates_applied_n = 0
    original_realms_preserved_n = 0
    coordinate_flag_disagreements_n = 0
    predicted_records_n = 0
    automatic_update_records_n = 0

    confidence_counter = Counter()

    all_header_needed = True
    update_header_needed = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            args.master,
            usecols=usecols,
            dtype=dtype,
            chunksize=args.chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk_rows_n = len(chunk)

        input_row_numbers = np.arange(
            total_rows,
            total_rows + chunk_rows_n,
            dtype=np.int64,
        )

        chunk["BTseq_id"] = normalise_text(
            chunk["BTseq_id"]
        )
        chunk["db_id"] = normalise_text(
            chunk["db_id"]
        )
        chunk["ptp_species"] = normalise_text(
            chunk["ptp_species"]
        )
        original_realm = normalise_text(
            chunk["realm"]
        )

        # Apply the direct realm update table by physical zero-based row.
        direct_for_chunk = direct.reindex(
            input_row_numbers
        )

        has_direct_update = (
            direct_for_chunk["assigned_realm"]
            .notna()
            .to_numpy()
        )

        direct_in_chunk_n = int(
            has_direct_update.sum()
        )

        if direct_in_chunk_n:
            master_ids = (
                chunk["BTseq_id"]
                .fillna("<MISSING>")
                .astype(str)
                .to_numpy()
            )

            update_ids = (
                direct_for_chunk["BTseq_id"]
                .fillna("<MISSING>")
                .astype(str)
                .to_numpy()
            )

            mismatched = (
                has_direct_update
                & (master_ids != update_ids)
            )

            if mismatched.any():
                positions = np.flatnonzero(
                    mismatched
                )[:10]

                examples = [
                    {
                        "input_row_number": int(
                            input_row_numbers[position]
                        ),
                        "master_BTseq_id": master_ids[position],
                        "update_BTseq_id": update_ids[position],
                    }
                    for position in positions
                ]

                raise RuntimeError(
                    "BTseq_id mismatch while applying direct "
                    f"realm updates. Examples: {examples}"
                )

            direct_updates_matched_n += direct_in_chunk_n

        effective_realm = original_realm.copy()

        direct_assigned_realm = normalise_text(
            direct_for_chunk[
                "assigned_realm"
            ].reset_index(drop=True)
        )

        apply_direct = (
            has_direct_update
            & effective_realm.isna().to_numpy()
        )

        preserve_existing = (
            has_direct_update
            & effective_realm.notna().to_numpy()
        )

        if apply_direct.any():
            effective_realm.iloc[
                np.flatnonzero(apply_direct)
            ] = direct_assigned_realm.iloc[
                np.flatnonzero(apply_direct)
            ].to_numpy()

        direct_updates_applied_n += int(
            apply_direct.sum()
        )
        original_realms_preserved_n += int(
            preserve_existing.sum()
        )

        # Determine coordinate validity from actual numeric values.
        valid_final_coordinate = actual_valid_coordinate(
            chunk["final_latitude"],
            chunk["final_longitude"],
        )

        # Supplementary comparison with the stored flag.
        stored_valid_flag = parse_boolean(
            chunk["final_coordinate_valid"]
        )

        flag_known = stored_valid_flag.notna()

        coordinate_flag_disagreements_n += int(
            (
                flag_known
                & (
                    stored_valid_flag.astype("boolean")
                    != pd.Series(
                        valid_final_coordinate,
                        index=chunk.index,
                        dtype="boolean",
                    )
                )
            ).fillna(False).sum()
        )

        # Formal inference targets:
        # - effective realm still missing;
        # - no valid final coordinate.
        inference_target = (
            effective_realm.isna().to_numpy()
            & ~valid_final_coordinate.to_numpy()
        )

        chunk_inference_targets_n = int(
            inference_target.sum()
        )

        inference_targets_n += (
            chunk_inference_targets_n
        )

        # Potential PTP:
        # - formal inference target;
        # - ptp_species exists;
        # - cluster has original-coordinate evidence.
        ptp_text = (
            chunk["ptp_species"]
            .fillna("")
            .astype(str)
        )

        cluster_has_evidence = (
            ptp_text.isin(candidate_cluster_names)
            .to_numpy()
        )

        potential_ptp = (
            inference_target
            & chunk["ptp_species"].notna().to_numpy()
            & cluster_has_evidence
        )

        chunk_potential_ptp_n = int(
            potential_ptp.sum()
        )

        potential_ptp_n += (
            chunk_potential_ptp_n
        )

        if chunk_potential_ptp_n:
            selected_positions = np.flatnonzero(
                potential_ptp
            )

            records = pd.DataFrame(
                {
                    "input_row_number":
                        input_row_numbers[
                            selected_positions
                        ],
                    "BTseq_id":
                        chunk["BTseq_id"].iloc[
                            selected_positions
                        ].reset_index(drop=True),
                    "db_id":
                        chunk["db_id"].iloc[
                            selected_positions
                        ].reset_index(drop=True),
                    "ptp_species":
                        chunk["ptp_species"].iloc[
                            selected_positions
                        ].reset_index(drop=True),
                }
            )

            records = records.merge(
                cluster_predictions,
                on="ptp_species",
                how="left",
                validate="many_to_one",
            )

            if records[
                "ptp_realm_confidence"
            ].isna().any():
                raise RuntimeError(
                    "Some Potential PTP records failed to "
                    "match the cluster prediction table."
                )

            records[
                "ptp_prediction_method"
            ] = "PTP_cluster_realm_inference"

            predicted_records_n += int(
                records["ptp_predicted_realm"]
                .notna()
                .sum()
            )

            confidence_counter.update(
                records["ptp_realm_confidence"]
                .astype(str)
                .tolist()
            )

            all_columns = [
                "input_row_number",
                "BTseq_id",
                "db_id",
                "ptp_species",
                "ptp_predicted_realm",
                "ptp_realm_confidence",
                "original_coordinate_locations_n",
                "assigned_realm_locations_n",
                "dominant_realm_locations_n",
                "dominant_realm_proportion",
                "realms_detected_n",
                "dominant_tied_realms_n",
                "realm_overlay_coverage",
                "ptp_auto_assignment_eligible",
                "ptp_prediction_method",
            ]

            append_csv(
                records[all_columns],
                all_temp,
                all_header_needed,
            )

            all_header_needed = False

            automatic = records.loc[
                records[
                    "ptp_auto_assignment_eligible"
                ]
                & records[
                    "ptp_predicted_realm"
                ].notna()
            ].copy()

            if not automatic.empty:
                automatic[
                    "assigned_realm"
                ] = automatic[
                    "ptp_predicted_realm"
                ]

                automatic[
                    "realm_assignment_method"
                ] = "PTP_cluster_realm_inference"

                automatic[
                    "realm_confidence"
                ] = automatic[
                    "ptp_realm_confidence"
                ]

                update_columns = [
                    "input_row_number",
                    "BTseq_id",
                    "db_id",
                    "ptp_species",
                    "assigned_realm",
                    "realm_assignment_method",
                    "realm_confidence",
                    "original_coordinate_locations_n",
                    "assigned_realm_locations_n",
                    "dominant_realm_locations_n",
                    "dominant_realm_proportion",
                    "realms_detected_n",
                    "realm_overlay_coverage",
                ]

                append_csv(
                    automatic[update_columns],
                    update_temp,
                    update_header_needed,
                )

                update_header_needed = False

                automatic_update_records_n += len(
                    automatic
                )

        total_rows += chunk_rows_n

        print(
            f"Chunk {chunk_number}: "
            f"rows={chunk_rows_n:,} | "
            f"inference targets="
            f"{chunk_inference_targets_n:,} | "
            f"Potential PTP="
            f"{chunk_potential_ptp_n:,} | "
            f"cumulative rows={total_rows:,}"
        )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("FINAL VALIDATION")
    print("=" * 100)

    print(f"Master rows:                  {total_rows:,}")
    print(
        "Direct updates matched:       "
        f"{direct_updates_matched_n:,}"
    )
    print(
        "Direct updates applied:       "
        f"{direct_updates_applied_n:,}"
    )
    print(
        "Existing realms preserved:    "
        f"{original_realms_preserved_n:,}"
    )
    print(
        "Coordinate flag differences:  "
        f"{coordinate_flag_disagreements_n:,}"
    )
    print(
        "Formal inference targets:     "
        f"{inference_targets_n:,}"
    )
    print(
        "Potential PTP records:         "
        f"{potential_ptp_n:,}"
    )
    print(
        "Records with PTP prediction:  "
        f"{predicted_records_n:,}"
    )
    print(
        "High/Medium update records:   "
        f"{automatic_update_records_n:,}"
    )

    if direct_updates_matched_n != len(direct):
        raise RuntimeError(
            "Not every direct update row was encountered in "
            "the master file. Expected "
            f"{len(direct):,}, found "
            f"{direct_updates_matched_n:,}."
        )

    count_errors = []

    if total_rows != EXPECTED_MASTER_ROWS:
        count_errors.append(
            "Master row count: "
            f"expected {EXPECTED_MASTER_ROWS:,}, "
            f"observed {total_rows:,}"
        )

    if inference_targets_n != EXPECTED_INFERENCE_TARGETS:
        count_errors.append(
            "Formal inference targets: "
            f"expected {EXPECTED_INFERENCE_TARGETS:,}, "
            f"observed {inference_targets_n:,}"
        )

    if potential_ptp_n != EXPECTED_POTENTIAL_PTP:
        count_errors.append(
            "Potential PTP records: "
            f"expected {EXPECTED_POTENTIAL_PTP:,}, "
            f"observed {potential_ptp_n:,}"
        )

    if count_errors and not args.allow_count_mismatch:
        print("\nCOUNT VALIDATION FAILED:")
        for error in count_errors:
            print(f"  - {error}")

        print(
            "\nTemporary record-level files were not "
            "finalized:"
        )
        print(f"  {all_temp}")
        print(f"  {update_temp}")

        raise RuntimeError(
            "Expected counts were not reproduced. "
            "Investigate before using PTP updates. "
            "Use --allow-count-mismatch only after the "
            "difference has been explained."
        )

    if count_errors:
        print("\nWARNING: finalizing despite count differences:")
        for error in count_errors:
            print(f"  - {error}")

    # If one confidence category produces no automatic assignments,
    # ensure the update file still exists with the expected columns.
    if update_header_needed:
        empty_update = pd.DataFrame(
            columns=[
                "input_row_number",
                "BTseq_id",
                "db_id",
                "ptp_species",
                "assigned_realm",
                "realm_assignment_method",
                "realm_confidence",
                "original_coordinate_locations_n",
                "assigned_realm_locations_n",
                "dominant_realm_locations_n",
                "dominant_realm_proportion",
                "realms_detected_n",
                "realm_overlay_coverage",
            ]
        )
        empty_update.to_csv(
            update_temp,
            index=False,
        )

    if all_header_needed:
        raise RuntimeError(
            "No Potential PTP records were produced."
        )

    os.replace(all_temp, all_output)
    os.replace(update_temp, update_output)

    # -------------------------------------------------------------------------
    # Summaries and configuration
    # -------------------------------------------------------------------------

    summary_rows = [
        {
            "summary_type": "overall",
            "value": "master_rows",
            "records_n": total_rows,
        },
        {
            "summary_type": "overall",
            "value": "direct_updates_matched",
            "records_n": direct_updates_matched_n,
        },
        {
            "summary_type": "overall",
            "value": "direct_updates_applied",
            "records_n": direct_updates_applied_n,
        },
        {
            "summary_type": "overall",
            "value": "formal_inference_targets",
            "records_n": inference_targets_n,
        },
        {
            "summary_type": "overall",
            "value": "potential_ptp_records",
            "records_n": potential_ptp_n,
        },
        {
            "summary_type": "overall",
            "value": "records_with_ptp_prediction",
            "records_n": predicted_records_n,
        },
        {
            "summary_type": "overall",
            "value": "automatic_high_medium_updates",
            "records_n": automatic_update_records_n,
        },
    ]

    for confidence in [
        "High",
        "Medium",
        "Low",
        "Insufficient",
    ]:
        records_n = int(
            confidence_counter.get(confidence, 0)
        )

        summary_rows.append(
            {
                "summary_type":
                    "record_confidence",
                "value": confidence,
                "records_n": records_n,
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary["percentage_of_potential_ptp"] = np.where(
        potential_ptp_n > 0,
        (
            summary["records_n"]
            / potential_ptp_n
            * 100
        ),
        np.nan,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    config = {
        "master": str(args.master),
        "evidence": str(args.evidence),
        "direct_updates": str(args.direct_updates),
        "output_dir": str(args.output_dir),
        "chunksize": args.chunksize,
        "input_row_number_definition":
            "physical_zero_based_master_csv_data_row",
        "direct_update_identity_check":
            "input_row_number_plus_BTseq_id",
        "ptp_output_identity_fields": [
            "input_row_number",
            "BTseq_id",
            "db_id",
        ],
        "coordinate_validity_definition":
            "numeric final latitude/longitude in valid bounds, "
            "excluding exactly 0,0",
        "prediction_weighting":
            "equal_weight_per_independent_original_coordinate_location",
        "valid_realm_assignment_statuses":
            sorted(VALID_REALM_STATUSES),
        "min_support": MIN_SUPPORT,
        "high_min_locations": HIGH_MIN_LOCATIONS,
        "high_purity": HIGH_PURITY,
        "medium_purity": MEDIUM_PURITY,
        "automatic_assignment_confidence": [
            "High",
            "Medium",
        ],
        "expected_master_rows":
            EXPECTED_MASTER_ROWS,
        "expected_inference_targets":
            EXPECTED_INFERENCE_TARGETS,
        "expected_potential_ptp":
            EXPECTED_POTENTIAL_PTP,
        "observed_master_rows":
            total_rows,
        "observed_inference_targets":
            inference_targets_n,
        "observed_potential_ptp":
            potential_ptp_n,
        "observed_automatic_updates":
            automatic_update_records_n,
        "cluster_summary":
            cluster_summary,
    }

    with open(
        config_output,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("FINAL RECORD CONFIDENCE")
    print("=" * 100)

    for confidence in [
        "High",
        "Medium",
        "Low",
        "Insufficient",
    ]:
        count = confidence_counter.get(
            confidence,
            0,
        )

        percentage = (
            count / potential_ptp_n * 100
            if potential_ptp_n
            else 0
        )

        print(
            f"{confidence:12s} "
            f"{count:>10,}  "
            f"({percentage:6.2f}%)"
        )

    print("\n" + "=" * 100)
    print("SAVED FILES")
    print("=" * 100)
    print(cluster_output)
    print(all_output)
    print(update_output)
    print(summary_output)
    print(config_output)

    print("\nPTP realm prediction completed successfully.")


if __name__ == "__main__":
    main()
