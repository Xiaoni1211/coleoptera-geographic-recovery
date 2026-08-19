#!/usr/bin/env python3
"""
Validate PTP-based realm recovery using the hidden locations from step 03.

This script:
1. removes every selected cluster-location key from the evidence;
2. summarises the remaining independent locations in each selected PTP cluster;
3. predicts the hidden location's realm from the remaining dominant realm;
4. assigns High / Medium / Low / Insufficient confidence; and
5. reports location-level and supplementary record-weighted accuracy.

Independent coordinate locations have equal weight in realm inference.
`records_at_location_n` is used only for a supplementary record-weighted
evaluation and never to choose the predicted realm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVIDENCE_FILENAME = "ptp_cluster_location_realm_evidence.csv"
HIDDEN_SAMPLE_FILENAME = "ptp_hidden_locations_sample.csv"
EXCLUSION_FILENAME = "ptp_hidden_location_exclusion_keys.csv"

RESULTS_FILENAME = "ptp_hidden_realm_validation_results.csv"
COMPOSITION_FILENAME = "ptp_hidden_remaining_cluster_realm_composition.csv"
OVERALL_FILENAME = "ptp_hidden_realm_validation_overall_summary.csv"
CONFIDENCE_FILENAME = "ptp_hidden_realm_validation_by_confidence.csv"
REALM_FILENAME = "ptp_hidden_realm_validation_by_true_realm.csv"
CONFUSION_FILENAME = "ptp_hidden_realm_validation_confusion_matrix.csv"
ERRORS_FILENAME = "ptp_hidden_realm_validation_errors.csv"
CONFIG_FILENAME = "ptp_hidden_realm_validation_config.json"

KEY_COLUMNS = [
    "ptp_species",
    "location_latitude",
    "location_longitude",
]

VALID_REALM_STATUSES = {
    "assigned",
    "assigned_multiple_polygons_same_realm",
}

EVIDENCE_REQUIRED_COLUMNS = {
    *KEY_COLUMNS,
    "records_at_location_n",
    "validation_true_realm",
    "realm_assignment_status",
}

HIDDEN_REQUIRED_COLUMNS = {
    "hidden_sample_id",
    *KEY_COLUMNS,
    "records_at_location_n",
    "validation_true_realm",
}

EXCLUSION_REQUIRED_COLUMNS = {
    "hidden_sample_id",
    *KEY_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exclude step-03 hidden locations, predict their realms from "
            "remaining PTP-cluster evidence, and evaluate accuracy."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="ptp_hidden_truth_validation",
        help=(
            "Directory containing steps 02 and 03 CSV files "
            "(default: ptp_hidden_truth_validation)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: the input directory).",
    )
    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=4,
        help=(
            "Decimal places used to identify independent locations "
            "(default: 4; must match steps 03 and 04)."
        ),
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=3,
        help=(
            "Minimum remaining realm-assigned independent locations required "
            "to make a prediction (default: 3)."
        ),
    )
    parser.add_argument(
        "--high-min-locations",
        type=int,
        default=5,
        help=(
            "Minimum remaining realm-assigned locations for High confidence "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--high-purity",
        type=float,
        default=0.90,
        help=(
            "Minimum dominant-realm proportion for High confidence "
            "(default: 0.90)."
        ),
    )
    parser.add_argument(
        "--medium-purity",
        type=float,
        default=0.70,
        help=(
            "Minimum dominant-realm proportion for Medium confidence "
            "(default: 0.70)."
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


def normalise_keys(
    frame: pd.DataFrame,
    coordinate_decimals: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["ptp_species"] = result["ptp_species"].astype("string").str.strip()

    for column in ("location_latitude", "location_longitude"):
        result[column] = pd.to_numeric(result[column], errors="coerce").round(
            coordinate_decimals
        )

    invalid = (
        result["ptp_species"].isna()
        | result["ptp_species"].eq("")
        | ~result["location_latitude"].between(-90, 90)
        | ~result["location_longitude"].between(-180, 180)
    )
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum()):,} rows with invalid cluster or "
            "coordinate keys."
        )

    return result


def clean_realms(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    invalid_values = {"", "N/A", "NA", "nan", "None", "<NA>"}
    return result.mask(result.isin(invalid_values))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def summarise_remaining_evidence(
    remaining_selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_locations = (
        remaining_selected.groupby("ptp_species", observed=True)
        .size()
        .rename("remaining_independent_locations_n")
    )

    assigned = remaining_selected.loc[
        remaining_selected["realm_assignment_status"].isin(
            VALID_REALM_STATUSES
        )
        & remaining_selected["validation_true_realm"].notna()
    ].copy()

    realm_counts = (
        assigned.groupby(
            ["ptp_species", "validation_true_realm"],
            observed=True,
        )
        .size()
        .rename("remaining_realm_locations_n")
        .reset_index()
        .rename(columns={"validation_true_realm": "evidence_realm"})
    )

    if realm_counts.empty:
        cluster_summary = total_locations.reset_index()
        cluster_summary["remaining_assigned_realm_locations_n"] = 0
        cluster_summary["remaining_unassigned_realm_locations_n"] = (
            cluster_summary["remaining_independent_locations_n"]
        )
        cluster_summary["remaining_realms_detected_n"] = 0
        cluster_summary["predicted_realm_candidate"] = pd.NA
        cluster_summary["dominant_realm_locations_n"] = 0
        cluster_summary["dominant_realm_proportion"] = np.nan
        cluster_summary["dominant_tied_realms_n"] = 0
        cluster_summary["realm_overlay_coverage"] = 0.0
        cluster_summary["remaining_realm_composition"] = "{}"
        return cluster_summary, realm_counts

    assigned_totals = (
        realm_counts.groupby("ptp_species", observed=True)[
            "remaining_realm_locations_n"
        ]
        .sum()
        .rename("remaining_assigned_realm_locations_n")
    )
    realms_detected = (
        realm_counts.groupby("ptp_species", observed=True)
        .size()
        .rename("remaining_realms_detected_n")
    )

    realm_counts["_maximum_count"] = realm_counts.groupby(
        "ptp_species",
        observed=True,
    )["remaining_realm_locations_n"].transform("max")

    tied_counts = (
        realm_counts.loc[
            realm_counts["remaining_realm_locations_n"].eq(
                realm_counts["_maximum_count"]
            )
        ]
        .groupby("ptp_species", observed=True)
        .size()
        .rename("dominant_tied_realms_n")
    )

    dominant = (
        realm_counts.sort_values(
            [
                "ptp_species",
                "remaining_realm_locations_n",
                "evidence_realm",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("ptp_species", keep="first")
        .set_index("ptp_species")
        .rename(
            columns={
                "evidence_realm": "predicted_realm_candidate",
                "remaining_realm_locations_n": (
                    "dominant_realm_locations_n"
                ),
            }
        )[
            [
                "predicted_realm_candidate",
                "dominant_realm_locations_n",
            ]
        ]
    )

    composition = (
        realm_counts.sort_values(
            ["ptp_species", "evidence_realm"],
            kind="mergesort",
        )
        .groupby("ptp_species", observed=True)
        .apply(
            lambda group: json.dumps(
                dict(
                    zip(
                        group["evidence_realm"],
                        group["remaining_realm_locations_n"].astype(int),
                    )
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
            include_groups=False,
        )
        .rename("remaining_realm_composition")
    )

    cluster_summary = pd.concat(
        [
            total_locations,
            assigned_totals,
            realms_detected,
            dominant,
            tied_counts,
            composition,
        ],
        axis=1,
    ).reset_index()

    integer_columns = [
        "remaining_independent_locations_n",
        "remaining_assigned_realm_locations_n",
        "remaining_realms_detected_n",
        "dominant_realm_locations_n",
        "dominant_tied_realms_n",
    ]
    for column in integer_columns:
        cluster_summary[column] = (
            pd.to_numeric(cluster_summary[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    cluster_summary["remaining_unassigned_realm_locations_n"] = (
        cluster_summary["remaining_independent_locations_n"]
        - cluster_summary["remaining_assigned_realm_locations_n"]
    )
    cluster_summary["dominant_realm_proportion"] = (
        cluster_summary["dominant_realm_locations_n"]
        / cluster_summary["remaining_assigned_realm_locations_n"].replace(
            0,
            np.nan,
        )
    )
    cluster_summary["realm_overlay_coverage"] = (
        cluster_summary["remaining_assigned_realm_locations_n"]
        / cluster_summary["remaining_independent_locations_n"].replace(
            0,
            np.nan,
        )
    )
    cluster_summary["remaining_realm_composition"] = cluster_summary[
        "remaining_realm_composition"
    ].fillna("{}")

    realm_counts = realm_counts.drop(columns="_maximum_count")
    return cluster_summary, realm_counts


def add_predictions(
    results: pd.DataFrame,
    min_support: int,
    high_min_locations: int,
    high_purity: float,
    medium_purity: float,
) -> pd.DataFrame:
    results = results.copy()

    numeric_fill_zero = [
        "remaining_independent_locations_n",
        "remaining_assigned_realm_locations_n",
        "remaining_unassigned_realm_locations_n",
        "remaining_realms_detected_n",
        "dominant_realm_locations_n",
        "dominant_tied_realms_n",
    ]
    for column in numeric_fill_zero:
        results[column] = (
            pd.to_numeric(results[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    results["remaining_realm_composition"] = results[
        "remaining_realm_composition"
    ].fillna("{}")
    results["realm_overlay_coverage"] = pd.to_numeric(
        results["realm_overlay_coverage"],
        errors="coerce",
    )
    results["dominant_realm_proportion"] = pd.to_numeric(
        results["dominant_realm_proportion"],
        errors="coerce",
    )

    no_evidence = results["remaining_independent_locations_n"].eq(0)
    insufficient = (
        results["remaining_assigned_realm_locations_n"].lt(min_support)
        & ~no_evidence
    )
    tied = (
        results["dominant_tied_realms_n"].gt(1)
        & ~no_evidence
        & ~insufficient
    )

    results["prediction_status"] = "predicted"
    results.loc[no_evidence, "prediction_status"] = "no_remaining_evidence"
    results.loc[
        insufficient,
        "prediction_status",
    ] = "insufficient_realm_support"
    results.loc[tied, "prediction_status"] = "ambiguous_dominant_realm_tie"

    predicted_mask = results["prediction_status"].eq("predicted")
    results["predicted_realm"] = results[
        "predicted_realm_candidate"
    ].where(predicted_mask)

    high = (
        predicted_mask
        & results["remaining_assigned_realm_locations_n"].ge(
            high_min_locations
        )
        & results["dominant_realm_proportion"].ge(high_purity)
    )
    medium = (
        predicted_mask
        & ~high
        & results["remaining_assigned_realm_locations_n"].ge(min_support)
        & results["dominant_realm_proportion"].ge(medium_purity)
    )
    low = predicted_mask & ~high & ~medium

    results["ptp_realm_confidence"] = "Insufficient"
    results.loc[low, "ptp_realm_confidence"] = "Low"
    results.loc[medium, "ptp_realm_confidence"] = "Medium"
    results.loc[high, "ptp_realm_confidence"] = "High"

    correct_values = np.where(
        predicted_mask,
        results["predicted_realm"].eq(
            results["validation_true_realm"]
        ),
        pd.NA,
    )
    results["realm_prediction_correct"] = pd.array(
        correct_values,
        dtype="boolean",
    )

    results["validation_outcome"] = "correct"
    results.loc[
        predicted_mask & ~results["realm_prediction_correct"].fillna(False),
        "validation_outcome",
    ] = "wrong_realm"
    results.loc[
        results["prediction_status"].eq("no_remaining_evidence"),
        "validation_outcome",
    ] = "not_predicted_no_remaining_evidence"
    results.loc[
        results["prediction_status"].eq("insufficient_realm_support"),
        "validation_outcome",
    ] = "not_predicted_insufficient_realm_support"
    results.loc[
        results["prediction_status"].eq(
            "ambiguous_dominant_realm_tie"
        ),
        "validation_outcome",
    ] = "not_predicted_ambiguous_tie"

    return results


def metric_rows(
    frame: pd.DataFrame,
) -> dict[str, int | float]:
    locations_n = int(len(frame))
    hidden_records_n = int(frame["records_at_location_n"].sum())
    predicted = frame["prediction_status"].eq("predicted")
    correct = frame["realm_prediction_correct"].fillna(False)

    predicted_locations_n = int(predicted.sum())
    correct_locations_n = int((predicted & correct).sum())
    incorrect_locations_n = int((predicted & ~correct).sum())

    predicted_records_n = int(
        frame.loc[predicted, "records_at_location_n"].sum()
    )
    correct_records_n = int(
        frame.loc[predicted & correct, "records_at_location_n"].sum()
    )

    return {
        "hidden_locations_n": locations_n,
        "predicted_locations_n": predicted_locations_n,
        "unpredicted_locations_n": locations_n - predicted_locations_n,
        "location_prediction_coverage": safe_ratio(
            predicted_locations_n,
            locations_n,
        ),
        "correct_predicted_locations_n": correct_locations_n,
        "incorrect_predicted_locations_n": incorrect_locations_n,
        "location_accuracy_among_predicted": safe_ratio(
            correct_locations_n,
            predicted_locations_n,
        ),
        "location_end_to_end_success_rate": safe_ratio(
            correct_locations_n,
            locations_n,
        ),
        "hidden_records_n": hidden_records_n,
        "predicted_records_n": predicted_records_n,
        "unpredicted_records_n": hidden_records_n - predicted_records_n,
        "record_prediction_coverage": safe_ratio(
            predicted_records_n,
            hidden_records_n,
        ),
        "correct_predicted_records_n": correct_records_n,
        "record_weighted_accuracy_among_predicted": safe_ratio(
            correct_records_n,
            predicted_records_n,
        ),
        "record_weighted_end_to_end_success_rate": safe_ratio(
            correct_records_n,
            hidden_records_n,
        ),
    }


def grouped_metrics(
    results: pd.DataFrame,
    group_column: str,
    order: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for group_value, group in results.groupby(
        group_column,
        observed=True,
        dropna=False,
    ):
        row = {group_column: group_value}
        row.update(metric_rows(group))
        rows.append(row)

    output = pd.DataFrame(rows)
    if order is not None:
        output["_sort_order"] = (
            output[group_column]
            .map({value: index for index, value in enumerate(order)})
            .fillna(len(order))
        )
        output = output.sort_values(
            ["_sort_order", group_column],
            kind="mergesort",
        ).drop(columns="_sort_order")
    else:
        output = output.sort_values(group_column, kind="mergesort")

    return output.reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if args.coordinate_decimals < 0:
        raise ValueError("--coordinate-decimals cannot be negative.")
    if args.min_support < 1:
        raise ValueError("--min-support must be at least 1.")
    if args.high_min_locations < args.min_support:
        raise ValueError(
            "--high-min-locations must be greater than or equal to "
            "--min-support."
        )
    if not (
        0 <= args.medium_purity <= args.high_purity <= 1
    ):
        raise ValueError(
            "Purity thresholds must satisfy "
            "0 <= medium-purity <= high-purity <= 1."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = input_dir / EVIDENCE_FILENAME
    hidden_path = input_dir / HIDDEN_SAMPLE_FILENAME
    exclusion_path = input_dir / EXCLUSION_FILENAME

    for path in (evidence_path, hidden_path, exclusion_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    print("=" * 100)
    print("PASS 1: LOAD AND VALIDATE INPUTS")
    print("=" * 100)
    print(f"Evidence:   {evidence_path}")
    print(f"Truth:      {hidden_path}")
    print(f"Exclusions: {exclusion_path}")
    print(
        "Confidence: "
        f"High = assigned locations >= {args.high_min_locations} and "
        f"purity >= {args.high_purity:.2f}; "
        f"Medium = assigned locations >= {args.min_support} and "
        f"purity >= {args.medium_purity:.2f}; Low otherwise"
    )

    evidence = pd.read_csv(evidence_path, low_memory=False)
    hidden = pd.read_csv(hidden_path, low_memory=False)
    exclusions = pd.read_csv(exclusion_path, low_memory=False)

    require_columns(evidence, EVIDENCE_REQUIRED_COLUMNS, evidence_path)
    require_columns(hidden, HIDDEN_REQUIRED_COLUMNS, hidden_path)
    require_columns(exclusions, EXCLUSION_REQUIRED_COLUMNS, exclusion_path)

    evidence = normalise_keys(evidence, args.coordinate_decimals)
    hidden = normalise_keys(hidden, args.coordinate_decimals)
    exclusions = normalise_keys(exclusions, args.coordinate_decimals)

    evidence["validation_true_realm"] = clean_realms(
        evidence["validation_true_realm"]
    )
    hidden["validation_true_realm"] = clean_realms(
        hidden["validation_true_realm"]
    )
    evidence["records_at_location_n"] = (
        pd.to_numeric(
            evidence["records_at_location_n"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    hidden["records_at_location_n"] = (
        pd.to_numeric(
            hidden["records_at_location_n"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    for frame_name, frame in (
        ("evidence", evidence),
        ("hidden sample", hidden),
        ("exclusion list", exclusions),
    ):
        duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
        if duplicated.any():
            raise ValueError(
                f"The {frame_name} contains {int(duplicated.sum()):,} rows "
                "with duplicated cluster-location keys."
            )

    if hidden["hidden_sample_id"].duplicated().any():
        raise ValueError("hidden_sample_id is not unique in the hidden sample.")
    if exclusions["hidden_sample_id"].duplicated().any():
        raise ValueError(
            "hidden_sample_id is not unique in the exclusion list."
        )
    if hidden["validation_true_realm"].isna().any():
        raise ValueError(
            "At least one hidden location has no validation_true_realm."
        )

    hidden_key_check = hidden[
        ["hidden_sample_id", *KEY_COLUMNS]
    ].merge(
        exclusions[["hidden_sample_id", *KEY_COLUMNS]],
        on=["hidden_sample_id", *KEY_COLUMNS],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not hidden_key_check["_merge"].eq("both").all():
        counts = hidden_key_check["_merge"].value_counts().to_dict()
        raise ValueError(
            "Hidden sample and exclusion list do not contain identical keys: "
            f"{counts}"
        )

    evidence_match = evidence.merge(
        exclusions[KEY_COLUMNS].assign(_hidden_key=True),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    hidden_evidence_mask = evidence_match["_hidden_key"].eq(True)
    matched_hidden_n = int(hidden_evidence_mask.sum())
    if matched_hidden_n != len(exclusions):
        raise ValueError(
            f"Only {matched_hidden_n:,} of {len(exclusions):,} exclusion "
            "keys matched the evidence file. Check coordinate precision and "
            "input files."
        )

    print(f"Evidence cluster-location rows: {len(evidence):,}")
    print(f"Hidden locations:              {len(hidden):,}")
    print(f"Matched hidden evidence rows:  {matched_hidden_n:,}")

    print("\n" + "=" * 100)
    print("PASS 2: REMOVE ALL HIDDEN CLUSTER-LOCATION EVIDENCE")
    print("=" * 100)

    remaining = evidence_match.loc[
        ~hidden_evidence_mask
    ].drop(columns="_hidden_key")

    leakage_check = remaining.merge(
        exclusions[KEY_COLUMNS],
        on=KEY_COLUMNS,
        how="inner",
    )
    if not leakage_check.empty:
        raise RuntimeError(
            f"Leakage check failed: {len(leakage_check):,} hidden locations "
            "remain in the evidence."
        )

    selected_clusters = hidden["ptp_species"].drop_duplicates()
    remaining_selected = remaining.loc[
        remaining["ptp_species"].isin(selected_clusters)
    ].copy()

    print(f"Hidden evidence locations removed: {matched_hidden_n:,}")
    print(f"Leakage locations remaining:      {len(leakage_check):,}")
    print(
        f"Remaining locations in selected clusters: "
        f"{len(remaining_selected):,}"
    )

    print("\n" + "=" * 100)
    print("PASS 3: SUMMARISE REMAINING REALM EVIDENCE")
    print("=" * 100)

    cluster_summary, realm_composition = summarise_remaining_evidence(
        remaining_selected
    )

    results = hidden.merge(
        cluster_summary,
        on="ptp_species",
        how="left",
        validate="one_to_one",
        suffixes=("", "_remaining"),
    )
    results = add_predictions(
        results,
        min_support=args.min_support,
        high_min_locations=args.high_min_locations,
        high_purity=args.high_purity,
        medium_purity=args.medium_purity,
    )

    print("\nPrediction status:")
    print(results["prediction_status"].value_counts(dropna=False).to_string())
    print("\nConfidence:")
    print(
        results["ptp_realm_confidence"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n" + "=" * 100)
    print("PASS 4: CALCULATE VALIDATION METRICS")
    print("=" * 100)

    overall = metric_rows(results)
    overall.update(
        {
            "hidden_keys_matched_to_evidence_n": matched_hidden_n,
            "hidden_keys_remaining_after_exclusion_n": int(
                len(leakage_check)
            ),
            "selected_ptp_clusters_n": int(
                results["ptp_species"].nunique()
            ),
        }
    )
    overall_table = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in overall.items()]
    )

    by_confidence = grouped_metrics(
        results,
        "ptp_realm_confidence",
        order=["High", "Medium", "Low", "Insufficient"],
    )
    by_realm = grouped_metrics(results, "validation_true_realm")

    confusion = pd.crosstab(
        results["validation_true_realm"],
        results["predicted_realm"].fillna("UNPREDICTED"),
        rownames=["validation_true_realm"],
        colnames=["predicted_realm"],
        dropna=False,
    ).reset_index()

    error_results = results.loc[
        ~results["realm_prediction_correct"].fillna(False)
    ].copy()

    results_path = output_dir / RESULTS_FILENAME
    composition_path = output_dir / COMPOSITION_FILENAME
    overall_path = output_dir / OVERALL_FILENAME
    confidence_path = output_dir / CONFIDENCE_FILENAME
    realm_path = output_dir / REALM_FILENAME
    confusion_path = output_dir / CONFUSION_FILENAME
    errors_path = output_dir / ERRORS_FILENAME
    config_path = output_dir / CONFIG_FILENAME

    results.to_csv(results_path, index=False)
    realm_composition.to_csv(composition_path, index=False)
    overall_table.to_csv(overall_path, index=False)
    by_confidence.to_csv(confidence_path, index=False)
    by_realm.to_csv(realm_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    error_results.to_csv(errors_path, index=False)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "coordinate_decimals": args.coordinate_decimals,
        "min_support": args.min_support,
        "high_min_locations": args.high_min_locations,
        "high_purity": args.high_purity,
        "medium_purity": args.medium_purity,
        "prediction_weighting": "equal_weight_per_independent_location",
        "supplementary_accuracy_weighting": (
            "records_at_hidden_location"
        ),
        "valid_realm_assignment_statuses": sorted(VALID_REALM_STATUSES),
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nOverall validation:")
    display_metrics = [
        "hidden_locations_n",
        "predicted_locations_n",
        "location_prediction_coverage",
        "correct_predicted_locations_n",
        "incorrect_predicted_locations_n",
        "location_accuracy_among_predicted",
        "location_end_to_end_success_rate",
        "hidden_records_n",
        "record_prediction_coverage",
        "record_weighted_accuracy_among_predicted",
    ]
    print(
        overall_table.loc[
            overall_table["metric"].isin(display_metrics)
        ].to_string(index=False)
    )

    print("\nBy confidence:")
    print(by_confidence.to_string(index=False))

    print("\nBy true realm:")
    print(by_realm.to_string(index=False))

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Hidden locations evaluated:       {len(results):,}")
    print(
        f"Predicted locations:              "
        f"{overall['predicted_locations_n']:,}"
    )
    print(
        f"Location prediction coverage:     "
        f"{overall['location_prediction_coverage']:.4f}"
    )
    print(
        f"Location accuracy when predicted: "
        f"{overall['location_accuracy_among_predicted']:.4f}"
    )
    print(
        f"End-to-end location success:      "
        f"{overall['location_end_to_end_success_rate']:.4f}"
    )
    print(
        f"Record-weighted accuracy:         "
        f"{overall['record_weighted_accuracy_among_predicted']:.4f}"
    )
    print(f"Hidden-location leakage remaining: {len(leakage_check):,}")
    print("\nSaved files:")
    for path in (
        results_path,
        composition_path,
        overall_path,
        confidence_path,
        realm_path,
        confusion_path,
        errors_path,
        config_path,
    ):
        print(path)
    print(
        "\nValidation completed. The source evidence and original dataset were "
        "not modified."
    )


if __name__ == "__main__":
    main()
