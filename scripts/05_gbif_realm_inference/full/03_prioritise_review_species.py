#!/usr/bin/env python3

"""
Prioritise GBIF species names requiring manual review.

The priority is based on the maximum number of additional candidate records
that GBIF species-range inference could potentially cover after accounting
for records already eligible for automatic PTP assignment.

This script does not call the GBIF API and does not modify the master dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REVIEW = Path(
    "gbif_full_prediction/"
    "gbif_species_name_matching/"
    "gbif_species_name_review_needed.csv"
)

DEFAULT_ACCEPTED = Path(
    "gbif_full_prediction/"
    "gbif_species_name_matching/"
    "gbif_species_name_accepted.csv"
)

DEFAULT_CANDIDATE_SPECIES = Path(
    "gbif_full_prediction/"
    "gbif_full_candidate_species.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "gbif_full_prediction/"
    "gbif_review_prioritisation"
)


COUNT_COLUMNS = [
    "candidate_records_n",
    "ptp_high_records_n",
    "ptp_medium_records_n",
    "ptp_low_records_n",
    "ptp_insufficient_records_n",
    "no_ptp_candidate_records_n",
    "ptp_auto_assignment_records_n",
]


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} file not found:\n{path}"
        )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def clean_species(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )
    return cleaned


def convert_count_columns(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    df = df.copy()

    for column in COUNT_COLUMNS:
        numeric = pd.to_numeric(df[column], errors="coerce")

        invalid = numeric.isna() & df[column].notna()
        if invalid.any():
            examples = df.loc[invalid, column].head(10).tolist()
            raise ValueError(
                f"{label}: non-numeric values found in {column}: "
                f"{examples}"
            )

        numeric = numeric.fillna(0)

        if (numeric < 0).any():
            raise ValueError(
                f"{label}: negative values found in {column}"
            )

        if ((numeric % 1) != 0).any():
            raise ValueError(
                f"{label}: non-integer values found in {column}"
            )

        df[column] = numeric.astype("int64")

    return df


def normalise_taxon_key(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    missing = numeric.isna()
    if missing.any():
        examples = series.loc[missing].head(10).tolist()
        raise ValueError(
            f"{label}: missing or invalid gbif_occurrence_taxonKey values. "
            f"Examples: {examples}"
        )

    if ((numeric % 1) != 0).any():
        raise ValueError(
            f"{label}: non-integer gbif_occurrence_taxonKey values found."
        )

    return numeric.astype("int64").astype("string")


def classify_review_type(row: pd.Series) -> str:
    input_flag = str(row.get("input_name_flag", "")).strip().lower()
    rank = str(row.get("gbif_rank", "")).strip().upper()
    match_type = str(row.get("gbif_matchType", "")).strip().upper()

    if input_flag == "placeholder_or_higher_taxon":
        return "input_placeholder_or_higher_taxon"

    if rank == "SPECIES" and match_type == "FUZZY":
        return "species_level_fuzzy_match"

    if rank == "SPECIES":
        return "other_species_level_review"

    if rank in {
        "GENUS",
        "FAMILY",
        "ORDER",
        "CLASS",
        "PHYLUM",
        "KINGDOM",
    }:
        return "higher_rank_match_unsafe"

    return "other_review"


def suggested_action(review_type: str) -> str:
    mapping = {
        "species_level_fuzzy_match":
            "Check spelling, authorship or synonym; accept only if the species identity is confirmed.",

        "other_species_level_review":
            "Inspect the GBIF species-level match and alternatives before accepting.",

        "higher_rank_match_unsafe":
            "Find a valid accepted species name; do not use the genus or higher-rank taxonKey.",

        "input_placeholder_or_higher_taxon":
            "Exclude unless the original record can be resolved to a species.",

        "other_review":
            "Inspect the name match and GBIF alternatives manually.",
    }

    return mapping[review_type]


def assign_priority_band(
    incremental_records: pd.Series,
    cumulative_before: pd.Series,
) -> pd.Series:
    conditions = [
        incremental_records.eq(0),
        cumulative_before.lt(0.50),
        cumulative_before.lt(0.80),
        cumulative_before.lt(0.90),
        cumulative_before.lt(0.95),
    ]

    choices = [
        "P0_no_incremental_records",
        "P1_first_50_percent",
        "P2_50_to_80_percent",
        "P3_80_to_90_percent",
        "P4_90_to_95_percent",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="P5_remaining",
        ),
        index=incremental_records.index,
        dtype="string",
    )


def make_threshold_summary(
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = [10, 25, 50, 100, 250, 500, 1000, 2000, len(ranked)]
    thresholds = sorted(set(min(value, len(ranked)) for value in thresholds))

    total_incremental = int(
        ranked["incremental_candidate_records_n"].sum()
    )

    rows = []

    for threshold in thresholds:
        subset = ranked.head(threshold)

        incremental = int(
            subset["incremental_candidate_records_n"].sum()
        )

        rows.append(
            {
                "top_review_species_n": threshold,
                "candidate_records_n": int(
                    subset["candidate_records_n"].sum()
                ),
                "ptp_auto_assignment_records_n": int(
                    subset["ptp_auto_assignment_records_n"].sum()
                ),
                "incremental_candidate_records_n": incremental,
                "proportion_of_all_review_incremental_candidates": (
                    incremental / total_incremental
                    if total_incremental > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def make_target_summary(
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    total_incremental = int(
        ranked["incremental_candidate_records_n"].sum()
    )

    targets = [0.50, 0.80, 0.90, 0.95, 0.99]
    rows = []

    for target in targets:
        if total_incremental == 0:
            required_species = 0
            reached_records = 0
            reached_proportion = 0.0
        else:
            reached = ranked[
                ranked["cumulative_incremental_proportion"] >= target
            ]

            if reached.empty:
                required_species = len(ranked)
            else:
                required_species = int(
                    reached.iloc[0]["review_priority_rank"]
                )

            selected = ranked.head(required_species)
            reached_records = int(
                selected["incremental_candidate_records_n"].sum()
            )
            reached_proportion = reached_records / total_incremental

        rows.append(
            {
                "target_incremental_coverage": target,
                "review_species_required_n": required_species,
                "incremental_candidate_records_reached_n":
                    reached_records,
                "actual_incremental_coverage": reached_proportion,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prioritise full GBIF review species and prepare "
            "accepted taxon queries."
        )
    )

    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW,
        help=f"Default: {DEFAULT_REVIEW}",
    )

    parser.add_argument(
        "--accepted",
        type=Path,
        default=DEFAULT_ACCEPTED,
        help=f"Default: {DEFAULT_ACCEPTED}",
    )

    parser.add_argument(
        "--candidate-species",
        type=Path,
        default=DEFAULT_CANDIDATE_SPECIES,
        help=f"Default: {DEFAULT_CANDIDATE_SPECIES}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Default: {DEFAULT_OUTPUT_DIR}",
    )

    args = parser.parse_args()

    require_file(args.review, "Review-name")
    require_file(args.accepted, "Accepted-name")
    require_file(args.candidate_species, "Candidate-species")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("LOAD INPUT FILES")

    review = pd.read_csv(args.review, low_memory=False)
    accepted = pd.read_csv(args.accepted, low_memory=False)
    candidate_species = pd.read_csv(
        args.candidate_species,
        low_memory=False,
    )

    print(f"Review species:        {len(review):,}")
    print(f"Accepted species:      {len(accepted):,}")
    print(f"Candidate species:     {len(candidate_species):,}")

    review_required = [
        "species",
        *COUNT_COLUMNS,
        "query_name",
        "input_name_flag",
        "gbif_occurrence_taxonKey",
        "gbif_scientificName",
        "gbif_acceptedScientificName",
        "gbif_canonicalName",
        "gbif_rank",
        "gbif_status",
        "gbif_matchType",
        "gbif_confidence",
        "gbif_alternatives_n",
        "gbif_alternatives_json",
        "match_decision",
        "match_decision_reason",
    ]

    accepted_required = [
        "species",
        *COUNT_COLUMNS,
        "gbif_occurrence_taxonKey",
        "gbif_scientificName",
        "gbif_acceptedScientificName",
        "gbif_canonicalName",
        "gbif_rank",
        "gbif_status",
        "gbif_matchType",
        "match_decision",
    ]

    candidate_required = [
        "input_species_names",
        *COUNT_COLUMNS,
    ]

    require_columns(review, review_required, "Review table")
    require_columns(accepted, accepted_required, "Accepted table")
    require_columns(
        candidate_species,
        candidate_required,
        "Candidate-species table",
    )

    review["species"] = clean_species(review["species"])
    accepted["species"] = clean_species(accepted["species"])
    candidate_species["input_species_names"] = clean_species(
        candidate_species["input_species_names"]
    )

    for df, name_column, label in [
        (review, "species", "Review table"),
        (accepted, "species", "Accepted table"),
        (
            candidate_species,
            "input_species_names",
            "Candidate-species table",
        ),
    ]:
        if df[name_column].isna().any():
            raise ValueError(
                f"{label}: missing species names found."
            )

        duplicated = df[name_column].duplicated(keep=False)
        if duplicated.any():
            examples = (
                df.loc[duplicated, name_column]
                .drop_duplicates()
                .head(20)
                .tolist()
            )
            raise ValueError(
                f"{label}: duplicated species names found: {examples}"
            )

    review = convert_count_columns(review, "Review table")
    accepted = convert_count_columns(accepted, "Accepted table")
    candidate_species = convert_count_columns(
        candidate_species,
        "Candidate-species table",
    )

    print_header("VERIFY NAME SETS AND COUNTS")

    review_names = set(review["species"])
    accepted_names = set(accepted["species"])
    candidate_names = set(candidate_species["input_species_names"])

    overlap = review_names & accepted_names
    unmatched_decisions = candidate_names - review_names - accepted_names
    unexpected_decisions = (
        review_names | accepted_names
    ) - candidate_names

    print(f"Accepted/review overlap:          {len(overlap):,}")
    print(f"Candidates without decision:      {len(unmatched_decisions):,}")
    print(f"Decisions outside candidates:     {len(unexpected_decisions):,}")

    if overlap:
        raise ValueError(
            "Some species occur in both accepted and review tables."
        )

    if unmatched_decisions:
        examples = sorted(unmatched_decisions)[:20]
        raise ValueError(
            "Some candidate species have no accepted/review decision. "
            f"Examples: {examples}"
        )

    if unexpected_decisions:
        examples = sorted(unexpected_decisions)[:20]
        raise ValueError(
            "Some accepted/review species are absent from the candidate "
            f"summary. Examples: {examples}"
        )

    candidate_check = candidate_species.rename(
        columns={"input_species_names": "species"}
    )

    combined_decisions = pd.concat(
        [
            accepted[["species", *COUNT_COLUMNS]],
            review[["species", *COUNT_COLUMNS]],
        ],
        ignore_index=True,
    )

    comparison = candidate_check.merge(
        combined_decisions,
        on="species",
        how="inner",
        suffixes=("_candidate", "_match"),
        validate="one_to_one",
    )

    mismatch_rows = pd.Series(False, index=comparison.index)

    for column in COUNT_COLUMNS:
        mismatch_rows |= (
            comparison[f"{column}_candidate"]
            != comparison[f"{column}_match"]
        )

    print(f"Species with count mismatch:      {mismatch_rows.sum():,}")

    if mismatch_rows.any():
        examples = comparison.loc[
            mismatch_rows,
            ["species"],
        ].head(20)

        raise ValueError(
            "Counts in name-matching tables do not agree with the "
            "candidate-species summary. Examples:\n"
            + examples.to_string(index=False)
        )

    for df, label in [
        (review, "Review"),
        (accepted, "Accepted"),
    ]:
        component_sum = (
            df["ptp_high_records_n"]
            + df["ptp_medium_records_n"]
            + df["ptp_low_records_n"]
            + df["ptp_insufficient_records_n"]
            + df["no_ptp_candidate_records_n"]
        )

        component_mismatch = (
            component_sum != df["candidate_records_n"]
        )

        auto_expected = (
            df["ptp_high_records_n"]
            + df["ptp_medium_records_n"]
        )

        auto_mismatch = (
            auto_expected
            != df["ptp_auto_assignment_records_n"]
        )

        print(
            f"{label} candidate accounting mismatch: "
            f"{component_mismatch.sum():,}"
        )
        print(
            f"{label} PTP auto-eligibility mismatch:  "
            f"{auto_mismatch.sum():,}"
        )

        if component_mismatch.any():
            raise ValueError(
                f"{label}: candidate-record components do not sum "
                "to candidate_records_n."
            )

        if auto_mismatch.any():
            raise ValueError(
                f"{label}: ptp_auto_assignment_records_n is not "
                "equal to PTP High + Medium."
            )

    print_header("BUILD REVIEW PRIORITY TABLE")

    review["incremental_candidate_records_n"] = (
        review["candidate_records_n"]
        - review["ptp_auto_assignment_records_n"]
    )

    if (review["incremental_candidate_records_n"] < 0).any():
        raise ValueError(
            "Negative incremental candidate-record counts found."
        )

    review["ptp_auto_coverage_proportion"] = np.where(
        review["candidate_records_n"] > 0,
        review["ptp_auto_assignment_records_n"]
        / review["candidate_records_n"],
        0.0,
    )

    review["review_type"] = review.apply(
        classify_review_type,
        axis=1,
    )

    review["suggested_review_action"] = (
        review["review_type"].map(suggested_action)
    )

    review = review.sort_values(
        by=[
            "incremental_candidate_records_n",
            "candidate_records_n",
            "species",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    review.insert(
        0,
        "review_priority_rank",
        np.arange(1, len(review) + 1),
    )

    total_incremental = int(
        review["incremental_candidate_records_n"].sum()
    )

    review["cumulative_incremental_candidate_records_n"] = (
        review["incremental_candidate_records_n"].cumsum()
    )

    if total_incremental > 0:
        review["cumulative_incremental_proportion"] = (
            review["cumulative_incremental_candidate_records_n"]
            / total_incremental
        )

        cumulative_before = (
            review["cumulative_incremental_candidate_records_n"]
            - review["incremental_candidate_records_n"]
        ) / total_incremental
    else:
        review["cumulative_incremental_proportion"] = 0.0
        cumulative_before = pd.Series(
            0.0,
            index=review.index,
        )

    review["priority_band"] = assign_priority_band(
        review["incremental_candidate_records_n"],
        cumulative_before,
    )

    preferred_columns = [
        "review_priority_rank",
        "priority_band",
        "species",
        "candidate_records_n",
        "ptp_high_records_n",
        "ptp_medium_records_n",
        "ptp_low_records_n",
        "ptp_insufficient_records_n",
        "no_ptp_candidate_records_n",
        "ptp_auto_assignment_records_n",
        "incremental_candidate_records_n",
        "ptp_auto_coverage_proportion",
        "cumulative_incremental_candidate_records_n",
        "cumulative_incremental_proportion",
        "review_type",
        "suggested_review_action",
        "query_name",
        "input_name_flag",
        "gbif_occurrence_taxonKey",
        "gbif_scientificName",
        "gbif_acceptedScientificName",
        "gbif_canonicalName",
        "gbif_rank",
        "gbif_status",
        "gbif_matchType",
        "gbif_confidence",
        "gbif_alternatives_n",
        "gbif_alternatives_json",
        "match_decision_reason",
    ]

    other_columns = [
        column for column in review.columns
        if column not in preferred_columns
    ]

    review = review[preferred_columns + other_columns]

    priority_path = (
        args.output_dir
        / "gbif_review_species_prioritised.csv"
    )

    review.to_csv(priority_path, index=False)

    print(f"Total review species:                 {len(review):,}")
    print(
        "Review candidate records:             "
        f"{review['candidate_records_n'].sum():,}"
    )
    print(
        "Already PTP High/Medium eligible:      "
        f"{review['ptp_auto_assignment_records_n'].sum():,}"
    )
    print(
        "Maximum incremental GBIF candidates:   "
        f"{total_incremental:,}"
    )
    print(
        "Review species with incremental value: "
        f"{(review['incremental_candidate_records_n'] > 0).sum():,}"
    )

    print_header("CREATE MANUAL REVIEW TEMPLATE")

    manual_columns = [
        "review_priority_rank",
        "priority_band",
        "species",
        "incremental_candidate_records_n",
        "candidate_records_n",
        "ptp_auto_assignment_records_n",
        "review_type",
        "query_name",
        "gbif_scientificName",
        "gbif_canonicalName",
        "gbif_rank",
        "gbif_matchType",
        "gbif_confidence",
        "gbif_occurrence_taxonKey",
        "gbif_alternatives_n",
        "gbif_alternatives_json",
        "suggested_review_action",
    ]

    manual = review.loc[
        review["incremental_candidate_records_n"] > 0,
        manual_columns,
    ].copy()

    manual["manual_decision"] = ""
    manual["manual_corrected_name"] = ""
    manual["manual_occurrence_taxonKey"] = ""
    manual["manual_notes"] = ""

    manual_path = (
        args.output_dir
        / "gbif_review_manual_decision_template.csv"
    )

    manual.to_csv(manual_path, index=False)

    print(f"Manual-review rows: {len(manual):,}")

    print_header("PREPARE ACCEPTED OCCURRENCE QUERY TABLES")

    accepted["gbif_occurrence_taxonKey"] = normalise_taxon_key(
        accepted["gbif_occurrence_taxonKey"],
        "Accepted table",
    )

    accepted["incremental_candidate_records_n"] = (
        accepted["candidate_records_n"]
        - accepted["ptp_auto_assignment_records_n"]
    )

    accepted_mapping_columns = [
        "species",
        "gbif_occurrence_taxonKey",
        "gbif_scientificName",
        "gbif_acceptedScientificName",
        "gbif_canonicalName",
        "gbif_rank",
        "gbif_status",
        "gbif_matchType",
        "match_decision",
        "candidate_records_n",
        "ptp_high_records_n",
        "ptp_medium_records_n",
        "ptp_low_records_n",
        "ptp_insufficient_records_n",
        "no_ptp_candidate_records_n",
        "ptp_auto_assignment_records_n",
        "incremental_candidate_records_n",
    ]

    accepted_mapping = accepted[
        accepted_mapping_columns
    ].sort_values(
        by=[
            "gbif_occurrence_taxonKey",
            "species",
        ],
        kind="stable",
    )

    accepted_mapping_path = (
        args.output_dir
        / "gbif_accepted_species_to_taxon_mapping.csv"
    )

    accepted_mapping.to_csv(
        accepted_mapping_path,
        index=False,
    )

    unique_queries = (
        accepted_mapping
        .groupby(
            "gbif_occurrence_taxonKey",
            as_index=False,
            sort=True,
        )
        .agg(
            gbif_scientificName=(
                "gbif_scientificName",
                "first",
            ),
            gbif_acceptedScientificName=(
                "gbif_acceptedScientificName",
                "first",
            ),
            gbif_canonicalName=(
                "gbif_canonicalName",
                "first",
            ),
            input_species_n=(
                "species",
                "nunique",
            ),
            input_species_names=(
                "species",
                lambda values: " | ".join(
                    sorted(set(values.dropna()))
                ),
            ),
            candidate_records_n=(
                "candidate_records_n",
                "sum",
            ),
            ptp_auto_assignment_records_n=(
                "ptp_auto_assignment_records_n",
                "sum",
            ),
            incremental_candidate_records_n=(
                "incremental_candidate_records_n",
                "sum",
            ),
        )
    )

    unique_queries = unique_queries.sort_values(
        by=[
            "incremental_candidate_records_n",
            "candidate_records_n",
            "gbif_occurrence_taxonKey",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    unique_queries.insert(
        0,
        "occurrence_query_priority",
        np.arange(1, len(unique_queries) + 1),
    )

    unique_queries_path = (
        args.output_dir
        / "gbif_accepted_unique_taxon_queries.csv"
    )

    unique_queries.to_csv(
        unique_queries_path,
        index=False,
    )

    print(f"Accepted input species:       {len(accepted_mapping):,}")
    print(f"Unique occurrence taxonKeys:  {len(unique_queries):,}")
    print(
        "Duplicated taxon mappings:   "
        f"{len(accepted_mapping) - len(unique_queries):,}"
    )

    print_header("CREATE SUMMARY TABLES")

    threshold_summary = make_threshold_summary(review)
    threshold_path = (
        args.output_dir
        / "gbif_review_top_n_coverage.csv"
    )
    threshold_summary.to_csv(threshold_path, index=False)

    target_summary = make_target_summary(review)
    target_path = (
        args.output_dir
        / "gbif_review_target_coverage.csv"
    )
    target_summary.to_csv(target_path, index=False)

    category_summary = (
        review
        .groupby(
            ["review_type", "priority_band"],
            dropna=False,
            as_index=False,
        )
        .agg(
            review_species_n=("species", "size"),
            candidate_records_n=(
                "candidate_records_n",
                "sum",
            ),
            ptp_auto_assignment_records_n=(
                "ptp_auto_assignment_records_n",
                "sum",
            ),
            incremental_candidate_records_n=(
                "incremental_candidate_records_n",
                "sum",
            ),
        )
        .sort_values(
            by=[
                "incremental_candidate_records_n",
                "review_species_n",
            ],
            ascending=[False, False],
        )
    )

    category_path = (
        args.output_dir
        / "gbif_review_category_summary.csv"
    )
    category_summary.to_csv(category_path, index=False)

    summary_rows = [
        {
            "metric": "candidate_species_total",
            "value": len(candidate_species),
        },
        {
            "metric": "accepted_species_total",
            "value": len(accepted),
        },
        {
            "metric": "review_species_total",
            "value": len(review),
        },
        {
            "metric": "accepted_unique_taxon_keys",
            "value": len(unique_queries),
        },
        {
            "metric": "review_candidate_records",
            "value": int(
                review["candidate_records_n"].sum()
            ),
        },
        {
            "metric": "review_ptp_auto_assignment_records",
            "value": int(
                review[
                    "ptp_auto_assignment_records_n"
                ].sum()
            ),
        },
        {
            "metric": "review_incremental_candidate_records",
            "value": total_incremental,
        },
        {
            "metric": "review_species_with_incremental_records",
            "value": int(
                (
                    review[
                        "incremental_candidate_records_n"
                    ] > 0
                ).sum()
            ),
        },
    ]

    summary = pd.DataFrame(summary_rows)
    summary_path = (
        args.output_dir
        / "gbif_review_prioritisation_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print()
    print("Top-N coverage:")
    print(threshold_summary.to_string(index=False))

    print()
    print("Species required to reach coverage targets:")
    print(target_summary.to_string(index=False))

    print_header("SAVED FILES")

    for path in [
        priority_path,
        manual_path,
        accepted_mapping_path,
        unique_queries_path,
        threshold_path,
        target_path,
        category_path,
        summary_path,
    ]:
        print(path)

    print()
    print("GBIF review prioritisation completed successfully.")


if __name__ == "__main__":
    main()
