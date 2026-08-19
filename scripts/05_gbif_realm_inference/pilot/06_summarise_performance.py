#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def wilson_interval(successes, total, z=1.96):
    """计算二项比例的95% Wilson置信区间。"""
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return centre - margin, centre + margin


def summarise_group(group):
    species_n = len(group)

    validation_records_n = int(
        group["validation_records_n"].sum()
    )
    correct_validation_records_n = int(
        group["correct_validation_records_n"].sum()
    )

    if validation_records_n > 0:
        weighted_record_match_rate = (
            correct_validation_records_n
            / validation_records_n
        )
    else:
        weighted_record_match_rate = np.nan

    record_ci_low, record_ci_high = wilson_interval(
        correct_validation_records_n,
        validation_records_n
    )

    dominant_matches = int(
        group["observed_dominant_realm_matches"].sum()
    )

    dominant_realm_accuracy = (
        dominant_matches / species_n
        if species_n > 0 else np.nan
    )

    dominant_ci_low, dominant_ci_high = wilson_interval(
        dominant_matches,
        species_n
    )

    return pd.Series({
        "species_n": species_n,
        "gbif_records_median": group[
            "gbif_assigned_realm_records_n"
        ].median(),
        "validation_records_n": validation_records_n,
        "correct_validation_records_n":
            correct_validation_records_n,
        "weighted_record_match_rate":
            weighted_record_match_rate,
        "record_match_rate_mean":
            group["record_match_rate"].mean(),
        "record_match_rate_median":
            group["record_match_rate"].median(),
        "record_match_rate_ci95_low": record_ci_low,
        "record_match_rate_ci95_high": record_ci_high,
        "dominant_realm_matches_n": dominant_matches,
        "dominant_realm_accuracy":
            dominant_realm_accuracy,
        "dominant_realm_accuracy_ci95_low":
            dominant_ci_low,
        "dominant_realm_accuracy_ci95_high":
            dominant_ci_high,
        "dominant_realm_proportion_median":
            group["dominant_realm_proportion"].median(),
        "overlay_coverage_median":
            group["overlay_coverage"].median()
    })


def evaluate_thresholds(df):
    """
    测试仅依赖GBIF预测阶段就能获得的指标：
    dominant_realm_proportion、GBIF记录数和overlay coverage。
    """
    results = []

    proportion_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    record_thresholds = [5, 10, 20, 50]
    coverage_thresholds = [0.70, 0.80, 0.90]

    for proportion_min in proportion_thresholds:
        for gbif_records_min in record_thresholds:
            for coverage_min in coverage_thresholds:

                selected = df[
                    (
                        df["dominant_realm_proportion"]
                        >= proportion_min
                    )
                    & (
                        df["gbif_assigned_realm_records_n"]
                        >= gbif_records_min
                    )
                    & (
                        df["overlay_coverage"]
                        >= coverage_min
                    )
                ].copy()

                if selected.empty:
                    continue

                total_validation = int(
                    selected["validation_records_n"].sum()
                )
                total_correct = int(
                    selected[
                        "correct_validation_records_n"
                    ].sum()
                )

                record_accuracy = (
                    total_correct / total_validation
                    if total_validation > 0 else np.nan
                )

                dominant_matches = int(
                    selected[
                        "observed_dominant_realm_matches"
                    ].sum()
                )

                dominant_accuracy = (
                    dominant_matches / len(selected)
                )

                results.append({
                    "dominant_realm_proportion_min":
                        proportion_min,
                    "gbif_assigned_realm_records_min":
                        gbif_records_min,
                    "overlay_coverage_min":
                        coverage_min,
                    "selected_species_n":
                        len(selected),
                    "selected_species_proportion":
                        len(selected) / len(df),
                    "validation_records_n":
                        total_validation,
                    "weighted_record_match_rate":
                        record_accuracy,
                    "dominant_realm_matches_n":
                        dominant_matches,
                    "dominant_realm_accuracy":
                        dominant_accuracy
                })

    result = pd.DataFrame(results)

    if not result.empty:
        result = result.sort_values(
            [
                "dominant_realm_accuracy",
                "weighted_record_match_rate",
                "selected_species_n"
            ],
            ascending=[False, False, False]
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarise GBIF species-range pilot validation "
            "performance and evaluate acceptance thresholds."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Species-level GBIF realm validation CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="gbif_species_range_pilot/"
                "gbif_pilot_performance",
        help="Output directory"
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_columns = [
        "species",
        "predicted_realm",
        "observed_dominant_realm",
        "pilot_realm_confidence",
        "dominant_realm_proportion",
        "gbif_assigned_realm_records_n",
        "overlay_coverage",
        "validation_records_n",
        "correct_validation_records_n",
        "record_match_rate",
        "observed_dominant_realm_matches"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Input file is missing required columns: "
            + ", ".join(missing)
        )

    numeric_columns = [
        "dominant_realm_proportion",
        "gbif_assigned_realm_records_n",
        "overlay_coverage",
        "validation_records_n",
        "correct_validation_records_n",
        "record_match_rate"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    boolean_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False
    }

    df["observed_dominant_realm_matches"] = (
        df["observed_dominant_realm_matches"]
        .astype(str)
        .str.lower()
        .map(boolean_map)
    )

    valid = df[
        df["validation_records_n"].fillna(0) > 0
    ].copy()

    valid = valid.dropna(
        subset=[
            "record_match_rate",
            "observed_dominant_realm_matches"
        ]
    )

    valid["observed_dominant_realm_matches"] = (
        valid["observed_dominant_realm_matches"]
        .astype(bool)
    )

    confidence_order = ["High", "Medium", "Low"]
    valid["pilot_realm_confidence"] = pd.Categorical(
        valid["pilot_realm_confidence"],
        categories=confidence_order,
        ordered=True
    )

    overall = summarise_group(valid).to_frame().T
    overall.insert(0, "performance_group", "Overall")

    by_confidence = (
        valid.groupby(
            "pilot_realm_confidence",
            observed=True,
            dropna=False
        )
        .apply(summarise_group)
        .reset_index()
        .rename(
            columns={
                "pilot_realm_confidence":
                    "performance_group"
            }
        )
    )

    performance = pd.concat(
        [overall, by_confidence],
        ignore_index=True
    )

    thresholds = evaluate_thresholds(valid)

    performance_path = (
        output_dir
        / "gbif_pilot_performance_by_confidence.csv"
    )
    thresholds_path = (
        output_dir
        / "gbif_pilot_acceptance_threshold_grid.csv"
    )
    valid_path = (
        output_dir
        / "gbif_pilot_validated_species_used.csv"
    )

    performance.to_csv(performance_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    valid.to_csv(valid_path, index=False)

    print("=" * 90)
    print("GBIF PILOT PERFORMANCE SUMMARY")
    print("=" * 90)
    print(f"Input rows:                    {len(df):,}")
    print(f"Validated species used:        {len(valid):,}")
    print()
    print(
        performance[
            [
                "performance_group",
                "species_n",
                "validation_records_n",
                "weighted_record_match_rate",
                "record_match_rate_median",
                "dominant_realm_accuracy"
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved files:")
    print(performance_path)
    print(thresholds_path)
    print(valid_path)

    if not thresholds.empty:
        print()
        print("Top threshold combinations:")
        print(
            thresholds.head(15).to_string(index=False)
        )


if __name__ == "__main__":
    main()
