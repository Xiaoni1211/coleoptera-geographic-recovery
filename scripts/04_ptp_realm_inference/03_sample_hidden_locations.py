#!/usr/bin/env python3
"""
Sample independent locations for PTP hidden-truth validation.

The script uses the outputs from 02_prepare_hidden_truth_pool.py. It does
not modify the original 1.8-million-row dataset and does not erase coordinates.
Instead, it creates a reproducible exclusion list. Downstream prediction must
remove these cluster-location combinations from the evidence before predicting
their realms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ELIGIBILITY_FILENAME = "ptp_hidden_truth_cluster_eligibility.csv"
EVIDENCE_FILENAME = "ptp_cluster_location_realm_evidence.csv"
OUTPUT_FILENAME = "ptp_hidden_locations_sample.csv"
EXCLUSION_FILENAME = "ptp_hidden_location_exclusion_keys.csv"
SUMMARY_FILENAME = "ptp_hidden_locations_sample_summary.csv"
CONFIG_FILENAME = "ptp_hidden_locations_sample_config.json"

ELIGIBILITY_REQUIRED_COLUMNS = {
    "ptp_species",
    "independent_locations_n",
    "eligible_for_hidden_validation",
}

EVIDENCE_REQUIRED_COLUMNS = {
    "ptp_species",
    "location_latitude",
    "location_longitude",
    "records_at_location_n",
    "validation_true_realm",
    "realm_assignment_status",
}

VALID_REALM_STATUSES = {
    "assigned",
    "assigned_multiple_polygons_same_realm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select independent PTP cluster locations for hidden-truth "
            "validation, with at most one hidden location per PTP cluster."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="ptp_hidden_truth_validation",
        help=(
            "Directory containing the two CSV files produced by step 02 "
            "(default: ptp_hidden_truth_validation)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: the input directory).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=1000,
        help="Number of independent locations to select (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--min-remaining-locations",
        type=int,
        default=3,
        help=(
            "Minimum independent locations that must remain in a cluster after "
            "one location is hidden (default: 3)."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("proportional", "balanced", "unstratified"),
        default="proportional",
        help=(
            "Realm allocation method. 'proportional' preserves the eligible "
            "candidate distribution; 'balanced' gives realms similar sample "
            "sizes where capacity allows; 'unstratified' samples clusters "
            "without realm quotas (default: proportional)."
        ),
    )
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    path: Path,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Available columns: {frame.columns.tolist()}"
        )


def parse_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalised = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = normalised.map(mapping)

    unexpected = sorted(
        normalised[normalised.notna() & parsed.isna()].drop_duplicates().tolist()
    )
    if unexpected:
        raise ValueError(
            "Unrecognised values in eligible_for_hidden_validation: "
            f"{unexpected}"
        )

    return parsed.fillna(False).astype(bool)


def largest_remainder_allocation(
    capacities: pd.Series,
    sample_n: int,
) -> pd.Series:
    """Allocate a proportional integer sample without exceeding capacity."""
    capacities = capacities.astype(int).sort_index()
    if sample_n > int(capacities.sum()):
        raise ValueError(
            f"Requested {sample_n:,} samples, but only "
            f"{int(capacities.sum()):,} candidates are available."
        )

    ideal = capacities / capacities.sum() * sample_n
    allocation = np.floor(ideal).astype(int)
    allocation = pd.Series(allocation, index=capacities.index)

    remaining = sample_n - int(allocation.sum())
    remainder_order = (
        (ideal - allocation)
        .sort_values(ascending=False, kind="mergesort")
        .index.tolist()
    )

    for realm in remainder_order:
        if remaining == 0:
            break
        if allocation.loc[realm] < capacities.loc[realm]:
            allocation.loc[realm] += 1
            remaining -= 1

    if remaining:
        spare = capacities - allocation
        for realm in spare.sort_values(ascending=False).index:
            take = min(int(spare.loc[realm]), remaining)
            allocation.loc[realm] += take
            remaining -= take
            if remaining == 0:
                break

    if remaining:
        raise RuntimeError("Could not complete proportional sample allocation.")

    return allocation.astype(int)


def balanced_allocation(
    capacities: pd.Series,
    sample_n: int,
) -> pd.Series:
    """Allocate as evenly as possible while respecting realm capacity."""
    capacities = capacities.astype(int).sort_index()
    if sample_n > int(capacities.sum()):
        raise ValueError(
            f"Requested {sample_n:,} samples, but only "
            f"{int(capacities.sum()):,} candidates are available."
        )

    allocation = pd.Series(0, index=capacities.index, dtype=int)
    remaining = sample_n

    while remaining:
        active = capacities.index[allocation < capacities].tolist()
        if not active:
            break

        base = max(1, remaining // len(active))
        changed = 0
        for realm in active:
            take = min(
                base,
                int(capacities.loc[realm] - allocation.loc[realm]),
                remaining,
            )
            allocation.loc[realm] += take
            remaining -= take
            changed += take
            if remaining == 0:
                break

        if changed == 0:
            break

    if remaining:
        raise RuntimeError("Could not complete balanced sample allocation.")

    return allocation


def choose_one_candidate_per_cluster(
    candidates: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Randomly choose one valid candidate location from each eligible cluster.

    Random numbers are assigned explicitly so the choice is reproducible and
    independent of the input CSV row order.
    """
    ordered = candidates.sort_values(
        ["ptp_species", "location_latitude", "location_longitude"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["_candidate_random"] = rng.random(len(ordered))

    chosen = (
        ordered.sort_values(
            ["ptp_species", "_candidate_random"],
            kind="mergesort",
        )
        .drop_duplicates("ptp_species", keep="first")
        .reset_index(drop=True)
    )
    return chosen


def sample_candidates(
    cluster_candidates: pd.DataFrame,
    sample_n: int,
    sampling_mode: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    capacities = (
        cluster_candidates["validation_true_realm"]
        .value_counts()
        .sort_index()
        .astype(int)
    )

    if sample_n > len(cluster_candidates):
        raise ValueError(
            f"Requested {sample_n:,} locations, but only "
            f"{len(cluster_candidates):,} eligible clusters have a valid "
            "realm-assigned candidate location."
        )

    if sampling_mode == "unstratified":
        selected_indices = rng.choice(
            cluster_candidates.index.to_numpy(),
            size=sample_n,
            replace=False,
        )
        selected = cluster_candidates.loc[selected_indices].copy()
        allocation = (
            selected["validation_true_realm"]
            .value_counts()
            .reindex(capacities.index, fill_value=0)
            .astype(int)
        )
        return selected, allocation

    if sampling_mode == "proportional":
        allocation = largest_remainder_allocation(capacities, sample_n)
    else:
        allocation = balanced_allocation(capacities, sample_n)

    selected_parts = []
    for realm, n_to_sample in allocation.items():
        if n_to_sample == 0:
            continue
        realm_rows = cluster_candidates.loc[
            cluster_candidates["validation_true_realm"].eq(realm)
        ]
        chosen_indices = rng.choice(
            realm_rows.index.to_numpy(),
            size=int(n_to_sample),
            replace=False,
        )
        selected_parts.append(cluster_candidates.loc[chosen_indices].copy())

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected, allocation


def main() -> None:
    args = parse_args()

    if args.sample_n <= 0:
        raise ValueError("--sample-n must be greater than zero.")
    if args.min_remaining_locations < 1:
        raise ValueError("--min-remaining-locations must be at least 1.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eligibility_path = input_dir / ELIGIBILITY_FILENAME
    evidence_path = input_dir / EVIDENCE_FILENAME

    if not eligibility_path.exists():
        raise FileNotFoundError(f"Eligibility file not found: {eligibility_path}")
    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence file not found: {evidence_path}")

    print("=" * 100)
    print("LOAD STEP-03 HIDDEN-TRUTH POOL")
    print("=" * 100)
    print(f"Eligibility: {eligibility_path}")
    print(f"Evidence:    {evidence_path}")
    print(f"Sample size: {args.sample_n:,} independent locations")
    print(f"Mode:        {args.sampling_mode}")
    print(f"Seed:        {args.seed}")

    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    evidence = pd.read_csv(evidence_path, low_memory=False)

    require_columns(
        eligibility,
        ELIGIBILITY_REQUIRED_COLUMNS,
        eligibility_path,
    )
    require_columns(evidence, EVIDENCE_REQUIRED_COLUMNS, evidence_path)

    eligibility = eligibility.copy()
    eligibility["eligible_for_hidden_validation"] = parse_boolean(
        eligibility["eligible_for_hidden_validation"]
    )
    eligibility["independent_locations_n"] = pd.to_numeric(
        eligibility["independent_locations_n"],
        errors="coerce",
    )

    minimum_before_hiding = args.min_remaining_locations + 1
    eligible_clusters = eligibility.loc[
        eligibility["eligible_for_hidden_validation"]
        & eligibility["independent_locations_n"].ge(minimum_before_hiding)
    ].copy()

    if eligible_clusters["ptp_species"].duplicated().any():
        duplicate_n = int(
            eligible_clusters["ptp_species"].duplicated(keep=False).sum()
        )
        raise ValueError(
            f"Eligibility file contains {duplicate_n:,} rows with duplicated "
            "eligible ptp_species values."
        )

    candidate_locations = evidence.merge(
        eligible_clusters,
        on="ptp_species",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_cluster"),
    )

    candidate_locations["location_latitude"] = pd.to_numeric(
        candidate_locations["location_latitude"],
        errors="coerce",
    )
    candidate_locations["location_longitude"] = pd.to_numeric(
        candidate_locations["location_longitude"],
        errors="coerce",
    )
    candidate_locations["validation_true_realm"] = (
        candidate_locations["validation_true_realm"]
        .astype("string")
        .str.strip()
    )

    candidate_locations = candidate_locations.loc[
        candidate_locations["realm_assignment_status"].isin(
            VALID_REALM_STATUSES
        )
        & candidate_locations["location_latitude"].between(-90, 90)
        & candidate_locations["location_longitude"].between(-180, 180)
        & candidate_locations["validation_true_realm"].notna()
        & ~candidate_locations["validation_true_realm"].isin(
            ["", "N/A", "NA", "nan", "None"]
        )
    ].copy()

    key_columns = [
        "ptp_species",
        "location_latitude",
        "location_longitude",
    ]
    duplicate_location_keys = candidate_locations.duplicated(
        key_columns,
        keep=False,
    )
    if duplicate_location_keys.any():
        duplicate_n = int(duplicate_location_keys.sum())
        raise ValueError(
            f"Found {duplicate_n:,} duplicated cluster-location rows after "
            "filtering; expected one row per PTP cluster and independent "
            "coordinate location."
        )

    if candidate_locations.empty:
        raise ValueError("No eligible realm-assigned locations remain.")

    rng = np.random.default_rng(args.seed)
    cluster_candidates = choose_one_candidate_per_cluster(
        candidate_locations,
        rng,
    )

    selected, allocation = sample_candidates(
        cluster_candidates,
        args.sample_n,
        args.sampling_mode,
        rng,
    )

    selected["_output_random"] = rng.random(len(selected))
    selected = selected.sort_values(
        "_output_random",
        kind="mergesort",
    ).reset_index(drop=True)
    selected.insert(
        0,
        "hidden_sample_id",
        [f"H{i:04d}" for i in range(1, len(selected) + 1)],
    )
    selected["remaining_independent_locations_n"] = (
        selected["independent_locations_n"] - 1
    )
    selected["hidden_sampling_mode"] = args.sampling_mode
    selected["hidden_sampling_seed"] = args.seed

    if len(selected) != args.sample_n:
        raise RuntimeError(
            f"Internal sampling error: selected {len(selected):,}, expected "
            f"{args.sample_n:,}."
        )
    if selected["ptp_species"].duplicated().any():
        raise RuntimeError(
            "Internal sampling error: more than one location was selected "
            "from at least one PTP cluster."
        )
    if selected["remaining_independent_locations_n"].lt(
        args.min_remaining_locations
    ).any():
        raise RuntimeError(
            "Internal sampling error: at least one selected location would "
            "leave insufficient cluster evidence."
        )

    internal_columns = ["_candidate_random", "_output_random"]
    selected = selected.drop(
        columns=[c for c in internal_columns if c in selected.columns]
    )

    preferred_columns = [
        "hidden_sample_id",
        "ptp_species",
        "location_latitude",
        "location_longitude",
        "records_at_location_n",
        "validation_true_realm",
        "realm_assignment_status",
        "realm_polygon_matches_n",
        "distinct_realms_matched_n",
        "independent_locations_n",
        "remaining_independent_locations_n",
        "original_coordinate_records_n",
        "assigned_realm_locations_n",
        "ambiguous_locations_n",
        "unmatched_locations_n",
        "dominant_realm",
        "dominant_realm_locations_n",
        "dominant_realm_proportion",
        "realms_detected_n",
        "species_names_n",
        "example_species",
        "eligible_hidden_locations_n",
        "hidden_sampling_mode",
        "hidden_sampling_seed",
    ]
    output_columns = [
        column for column in preferred_columns if column in selected.columns
    ]
    remaining_columns = [
        column for column in selected.columns if column not in output_columns
    ]
    selected = selected[output_columns + remaining_columns]

    sample_path = output_dir / OUTPUT_FILENAME
    exclusion_path = output_dir / EXCLUSION_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    config_path = output_dir / CONFIG_FILENAME

    selected.to_csv(sample_path, index=False)
    selected[
        [
            "hidden_sample_id",
            "ptp_species",
            "location_latitude",
            "location_longitude",
        ]
    ].to_csv(exclusion_path, index=False)

    capacity = (
        cluster_candidates["validation_true_realm"]
        .value_counts()
        .sort_index()
    )
    realised = (
        selected["validation_true_realm"]
        .value_counts()
        .reindex(capacity.index, fill_value=0)
        .astype(int)
    )
    summary = pd.DataFrame(
        {
            "validation_true_realm": capacity.index,
            "eligible_cluster_candidates_n": capacity.values,
            "allocated_hidden_locations_n": allocation.reindex(
                capacity.index,
                fill_value=0,
            ).astype(int).values,
            "selected_hidden_locations_n": realised.values,
        }
    )
    summary["selected_proportion"] = (
        summary["selected_hidden_locations_n"] / args.sample_n
    )
    summary.to_csv(summary_path, index=False)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "eligibility_file": str(eligibility_path),
        "evidence_file": str(evidence_path),
        "sample_n": args.sample_n,
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
        "min_remaining_locations": args.min_remaining_locations,
        "eligible_clusters_after_threshold_n": int(len(eligible_clusters)),
        "eligible_realm_assigned_locations_n": int(len(candidate_locations)),
        "eligible_clusters_with_candidate_location_n": int(
            len(cluster_candidates)
        ),
        "selected_hidden_locations_n": int(len(selected)),
        "selected_hidden_records_n": int(
            pd.to_numeric(
                selected["records_at_location_n"],
                errors="coerce",
            ).fillna(0).sum()
        ),
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("REALM-STRATIFIED SAMPLE")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(
        f"Eligible clusters after threshold:      "
        f"{len(eligible_clusters):,}"
    )
    print(
        f"Eligible realm-assigned locations:      "
        f"{len(candidate_locations):,}"
    )
    print(
        f"Clusters with a valid candidate:        "
        f"{len(cluster_candidates):,}"
    )
    print(f"Selected hidden locations:              {len(selected):,}")
    print(
        f"Underlying records at hidden locations: "
        f"{config['selected_hidden_records_n']:,}"
    )
    print(
        f"Minimum locations remaining:            "
        f"{int(selected['remaining_independent_locations_n'].min()):,}"
    )
    print(f"Unique selected PTP clusters:           {selected['ptp_species'].nunique():,}")
    print("\nSaved files:")
    print(sample_path)
    print(exclusion_path)
    print(summary_path)
    print(config_path)
    print(
        "\nSampling completed. No coordinates in the original dataset were "
        "modified. Use the exclusion-key file to remove the selected "
        "cluster-location evidence during prediction."
    )


if __name__ == "__main__":
    main()
