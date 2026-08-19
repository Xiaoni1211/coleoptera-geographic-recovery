#!/usr/bin/env python3

"""
01_match_species_names.py

Match 200 pilot species names against the GBIF Species API.

Input
-----
ptp_geographic_recovery_pilot/species_range_pilot_candidates.csv

Required input column
---------------------
species

Outputs
-------
gbif_species_range_pilot/
    gbif_species_name_matches_all.csv
    gbif_species_name_accepted.csv
    gbif_species_name_review_needed.csv
    gbif_species_name_failed.csv
    gbif_species_name_match_summary.csv
    gbif_species_name_match_checkpoint.csv

The checkpoint permits safe restart after interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"

DEFAULT_INPUT = (
    "ptp_geographic_recovery_pilot/"
    "species_range_pilot_candidates.csv"
)

DEFAULT_OUTPUT_DIR = "gbif_species_range_pilot"

REQUIRED_COLUMNS = [
    "species",
    "missing_records_n",
    "internal_original_coordinate_records_n",
    "internal_original_realms_n",
    "sampling_band",
]

OUTPUT_COLUMNS = [
    # Original pilot information
    "species",
    "missing_records_n",
    "internal_original_coordinate_records_n",
    "internal_original_realms_n",
    "sampling_band",

    # Input-name screening
    "query_name",
    "input_name_flag",
    "input_name_species_like",

    # API request status
    "api_request_status",
    "api_http_status",
    "api_error",

    # GBIF match
    "gbif_usageKey",
    "gbif_acceptedUsageKey",
    "gbif_occurrence_taxonKey",
    "gbif_scientificName",
    "gbif_acceptedScientificName",
    "gbif_canonicalName",
    "gbif_rank",
    "gbif_status",
    "gbif_matchType",
    "gbif_confidence",
    "gbif_kingdom",
    "gbif_phylum",
    "gbif_class",
    "gbif_order",
    "gbif_family",
    "gbif_genus",
    "gbif_note",
    "gbif_alternatives_n",
    "gbif_alternatives_json",

    # Final decision
    "match_decision",
    "match_decision_reason",
    "manual_review_required",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match the 200 species-range pilot names against "
            "the GBIF Species API."
        )
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input pilot CSV. Default: {DEFAULT_INPUT}",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.20,
        help="Delay in seconds between successful API requests.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each API request.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum attempts for each name.",
    )

    parser.add_argument(
        "--minimum-confidence",
        type=int,
        default=95,
        help=(
            "Minimum confidence for automatic acceptance of "
            "an EXACT species-level match."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore an existing checkpoint and query every name again. "
            "Existing final outputs will be replaced."
        ),
    )

    return parser.parse_args()


def normalise_name(value: Any) -> str:
    if pd.isna(value):
        return ""

    name = str(value).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def screen_input_name(name: str) -> tuple[str, bool]:
    """
    Screen obvious placeholders or uncertain identifications.

    This does not replace GBIF matching. Flagged names are still sent
    to GBIF, but they cannot be accepted automatically.
    """

    if not name:
        return "missing_name", False

    lower_name = name.lower()

    uncertain_pattern = re.compile(
        r"(^|\s)(cf|aff|nr)\.?(?=\s|$)",
        flags=re.IGNORECASE,
    )

    placeholder_pattern = re.compile(
        r"(^|\s)"
        r"(sp|spp|gen|genus|species|unknown|unidentified)"
        r"\.?(?=\s|$)",
        flags=re.IGNORECASE,
    )

    if uncertain_pattern.search(lower_name):
        return "uncertain_identification", False

    if placeholder_pattern.search(lower_name):
        return "placeholder_or_higher_taxon", False

    tokens = name.split()

    if len(tokens) < 2:
        return "not_binomial", False

    # A species name should normally start with:
    # Capitalised genus + lowercase specific epithet.
    genus = tokens[0]
    epithet = tokens[1]

    genus_ok = bool(
        re.fullmatch(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", genus)
    )

    epithet_ok = bool(
        re.fullmatch(r"[a-z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", epithet)
    )

    if not genus_ok or not epithet_ok:
        return "unusual_name_format", False

    if len(tokens) > 2:
        # Authorship or subspecies information may be present.
        # It is queried, but must be manually reviewed.
        return "additional_name_tokens", False

    return "clean_binomial", True


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def request_gbif_match(
    session: requests.Session,
    name: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    """
    Request a species-level match constrained to Animalia/Coleoptera.
    """

    parameters = {
        "name": name,
        "rank": "SPECIES",
        "kingdom": "Animalia",
        "order": "Coleoptera",
        "verbose": "true",
    }

    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                GBIF_MATCH_URL,
                params=parameters,
                timeout=timeout,
            )

            if response.status_code == 200:
                result = response.json()

                if not isinstance(result, dict):
                    raise ValueError(
                        "GBIF response was not a JSON object."
                    )

                return {
                    "request_status": "success",
                    "http_status": response.status_code,
                    "error": "",
                    "result": result,
                }

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

            # Retry server errors and rate limiting.
            if response.status_code == 429 or response.status_code >= 500:
                wait_seconds = min(2 ** (attempt - 1), 30)
                time.sleep(wait_seconds)
                continue

            return {
                "request_status": "failed",
                "http_status": response.status_code,
                "error": last_error,
                "result": {},
            }

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

            if attempt < max_retries:
                wait_seconds = min(2 ** (attempt - 1), 30)
                time.sleep(wait_seconds)

    return {
        "request_status": "failed",
        "http_status": None,
        "error": last_error,
        "result": {},
    }


def classify_match(
    input_flag: str,
    input_species_like: bool,
    api_status: str,
    gbif_result: dict[str, Any],
    minimum_confidence: int,
) -> tuple[str, str, bool]:
    """
    Return:
        match_decision
        match_decision_reason
        manual_review_required
    """

    if api_status != "success":
        return (
            "api_failed",
            "GBIF API request failed; retry is required.",
            True,
        )

    match_type = safe_text(
        gbif_result.get("matchType")
    ).upper()

    confidence = safe_int(
        gbif_result.get("confidence")
    )

    rank = safe_text(
        gbif_result.get("rank")
    ).upper()

    status = safe_text(
        gbif_result.get("status")
    ).upper()

    kingdom = safe_text(
        gbif_result.get("kingdom")
    ).lower()

    order = safe_text(
        gbif_result.get("order")
    ).lower()

    usage_key = safe_int(
        gbif_result.get("usageKey")
    )

    accepted_usage_key = safe_int(
        gbif_result.get("acceptedUsageKey")
    )

    if match_type in {"", "NONE"} or (
        usage_key is None and accepted_usage_key is None
    ):
        return (
            "no_match",
            "GBIF returned no usable taxon match.",
            True,
        )

    if not input_species_like:
        return (
            "review",
            f"Input name flag: {input_flag}.",
            True,
        )

    if match_type == "HIGHERRANK":
        return (
            "review",
            "GBIF matched only a higher taxonomic rank.",
            True,
        )

    if rank != "SPECIES":
        return (
            "review",
            f"GBIF matched rank is {rank or 'missing'}, not SPECIES.",
            True,
        )

    if kingdom and kingdom != "animalia":
        return (
            "review",
            f"Matched kingdom is {kingdom}, not Animalia.",
            True,
        )

    if not order:
        return (
            "review",
            "Matched order is missing and Coleoptera cannot be verified.",
            True,
        )

    if order != "coleoptera":
        return (
            "review",
            f"Matched order is {order}, not Coleoptera.",
            True,
        )

    if status not in {"ACCEPTED", "SYNONYM"}:
        return (
            "review",
            f"GBIF taxonomic status is {status or 'missing'}.",
            True,
        )

    if match_type != "EXACT":
        return (
            "review",
            f"GBIF match type is {match_type}, not EXACT.",
            True,
        )

    if confidence is None:
        return (
            "review",
            "GBIF confidence is missing.",
            True,
        )

    if confidence < minimum_confidence:
        return (
            "review",
            (
                f"GBIF confidence {confidence} is below "
                f"the automatic threshold {minimum_confidence}."
            ),
            True,
        )

    if status == "SYNONYM":
        return (
            "accepted_synonym",
            (
                "Exact high-confidence species-level Coleoptera match; "
                "GBIF treats the input name as a synonym."
            ),
            False,
        )

    return (
        "accepted_exact",
        (
            "Exact high-confidence accepted species-level "
            "Coleoptera match."
        ),
        False,
    )


def build_output_row(
    input_row: dict[str, Any],
    api_response: dict[str, Any],
    minimum_confidence: int,
) -> dict[str, Any]:
    name = normalise_name(input_row.get("species"))

    input_flag, species_like = screen_input_name(name)

    result = api_response.get("result", {})
    alternatives = result.get("alternatives", [])

    if not isinstance(alternatives, list):
        alternatives = []

    usage_key = safe_int(result.get("usageKey"))
    accepted_usage_key = safe_int(result.get("acceptedUsageKey"))

    # For accepted names, acceptedUsageKey is often absent.
    # For synonyms, prefer the acceptedUsageKey.
    occurrence_taxon_key = accepted_usage_key or usage_key

    decision, reason, manual_review = classify_match(
        input_flag=input_flag,
        input_species_like=species_like,
        api_status=api_response.get("request_status", "failed"),
        gbif_result=result,
        minimum_confidence=minimum_confidence,
    )

    output = {
        "species": input_row.get("species"),
        "missing_records_n": input_row.get("missing_records_n"),
        "internal_original_coordinate_records_n":
            input_row.get("internal_original_coordinate_records_n"),
        "internal_original_realms_n":
            input_row.get("internal_original_realms_n"),
        "sampling_band": input_row.get("sampling_band"),

        "query_name": name,
        "input_name_flag": input_flag,
        "input_name_species_like": species_like,

        "api_request_status":
            api_response.get("request_status", "failed"),
        "api_http_status": api_response.get("http_status"),
        "api_error": api_response.get("error", ""),

        "gbif_usageKey": usage_key,
        "gbif_acceptedUsageKey": accepted_usage_key,
        "gbif_occurrence_taxonKey": occurrence_taxon_key,
        "gbif_scientificName":
            result.get("scientificName"),
        "gbif_acceptedScientificName":
            result.get("acceptedScientificName"),
        "gbif_canonicalName":
            result.get("canonicalName"),
        "gbif_rank":
            result.get("rank"),
        "gbif_status":
            result.get("status"),
        "gbif_matchType":
            result.get("matchType"),
        "gbif_confidence":
            result.get("confidence"),
        "gbif_kingdom":
            result.get("kingdom"),
        "gbif_phylum":
            result.get("phylum"),
        "gbif_class":
            result.get("class"),
        "gbif_order":
            result.get("order"),
        "gbif_family":
            result.get("family"),
        "gbif_genus":
            result.get("genus"),
        "gbif_note":
            result.get("note"),
        "gbif_alternatives_n":
            len(alternatives),
        "gbif_alternatives_json":
            json.dumps(
                alternatives,
                ensure_ascii=False,
                separators=(",", ":"),
            ),

        "match_decision": decision,
        "match_decision_reason": reason,
        "manual_review_required": manual_review,
    }

    return output


def load_checkpoint(
    checkpoint_path: Path,
    restart: bool,
) -> pd.DataFrame:
    if restart or not checkpoint_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        checkpoint = pd.read_csv(
            checkpoint_path,
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read checkpoint {checkpoint_path}: {exc}"
        ) from exc

    for column in OUTPUT_COLUMNS:
        if column not in checkpoint.columns:
            checkpoint[column] = pd.NA

    return checkpoint[OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        output_dir / "gbif_species_name_match_checkpoint.csv"
    )

    all_output_path = (
        output_dir / "gbif_species_name_matches_all.csv"
    )

    accepted_output_path = (
        output_dir / "gbif_species_name_accepted.csv"
    )

    review_output_path = (
        output_dir / "gbif_species_name_review_needed.csv"
    )

    failed_output_path = (
        output_dir / "gbif_species_name_failed.csv"
    )

    summary_output_path = (
        output_dir / "gbif_species_name_match_summary.csv"
    )

    print("=" * 88)
    print("GBIF SPECIES NAME MATCHING")
    print("=" * 88)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum automatic confidence: {args.minimum_confidence}")
    print(f"Restart requested: {args.restart}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    pilot = pd.read_csv(input_path, low_memory=False)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in pilot.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input is missing required columns: {missing_columns}"
        )

    pilot = pilot[REQUIRED_COLUMNS].copy()
    pilot["species"] = pilot["species"].map(normalise_name)

    if pilot["species"].eq("").any():
        raise ValueError(
            "Input contains missing or empty species names."
        )

    duplicated = pilot["species"].duplicated(keep=False)

    if duplicated.any():
        duplicate_names = sorted(
            pilot.loc[duplicated, "species"].unique()
        )

        raise ValueError(
            "Input contains duplicated species names: "
            f"{duplicate_names[:20]}"
        )

    print(f"Input rows: {len(pilot):,}")
    print(f"Unique species: {pilot['species'].nunique():,}")

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        restart=args.restart,
    )

    completed_names = set(
        checkpoint.loc[
            checkpoint["api_request_status"].eq("success"),
            "query_name",
        ]
        .dropna()
        .astype(str)
    )

    if completed_names:
        print(
            f"Successful names already in checkpoint: "
            f"{len(completed_names):,}"
        )

    pending = pilot.loc[
        ~pilot["species"].isin(completed_names)
    ].copy()

    print(f"Names still requiring API request: {len(pending):,}")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "MSc-Coleoptera-GBIF-Realm-Pilot/1.0 "
                "(academic research)"
            ),
        }
    )

    new_rows: list[dict[str, Any]] = []

    for request_number, input_row in enumerate(
        pending.to_dict(orient="records"),
        start=1,
    ):
        name = normalise_name(input_row["species"])

        print(
            f"[{request_number:03d}/{len(pending):03d}] "
            f"{name}",
            end="",
            flush=True,
        )

        api_response = request_gbif_match(
            session=session,
            name=name,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

        output_row = build_output_row(
            input_row=input_row,
            api_response=api_response,
            minimum_confidence=args.minimum_confidence,
        )

        new_rows.append(output_row)

        print(
            f" -> {output_row['match_decision']} "
            f"| {output_row['gbif_scientificName']} "
            f"| confidence={output_row['gbif_confidence']}"
        )

        updated_checkpoint = pd.concat(
            [
                checkpoint,
                pd.DataFrame(new_rows),
            ],
            ignore_index=True,
        )

        # If a previously failed name was re-requested, retain the latest row.
        updated_checkpoint = (
            updated_checkpoint
            .drop_duplicates(
                subset=["query_name"],
                keep="last",
            )
        )

        updated_checkpoint = updated_checkpoint.reindex(
            columns=OUTPUT_COLUMNS
        )

        atomic_write_csv(
            updated_checkpoint,
            checkpoint_path,
        )

        if api_response["request_status"] == "success":
            time.sleep(max(args.sleep, 0))

    final = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )

    # Keep the same order as the original 200-species pilot.
    order_lookup = {
        name: position
        for position, name in enumerate(pilot["species"])
    }

    final["_input_order"] = (
        final["query_name"]
        .map(order_lookup)
        .fillna(len(order_lookup))
    )

    final = (
        final
        .sort_values("_input_order")
        .drop(columns="_input_order")
        .reset_index(drop=True)
    )

    final = final.reindex(columns=OUTPUT_COLUMNS)

    accepted = final.loc[
        final["match_decision"].isin(
            ["accepted_exact", "accepted_synonym"]
        )
    ].copy()

    review = final.loc[
        final["match_decision"].eq("review")
    ].copy()

    failed = final.loc[
        final["match_decision"].isin(
            ["no_match", "api_failed"]
        )
    ].copy()

    atomic_write_csv(final, all_output_path)
    atomic_write_csv(accepted, accepted_output_path)
    atomic_write_csv(review, review_output_path)
    atomic_write_csv(failed, failed_output_path)

    summary_parts = []

    decision_summary = (
        final.groupby(
            ["match_decision"],
            dropna=False,
        )
        .size()
        .reset_index(name="species_n")
    )

    decision_summary.insert(0, "summary_type", "match_decision")
    decision_summary["sampling_band"] = pd.NA

    summary_parts.append(
        decision_summary[
            [
                "summary_type",
                "sampling_band",
                "match_decision",
                "species_n",
            ]
        ]
    )

    band_summary = (
        final.groupby(
            ["sampling_band", "match_decision"],
            dropna=False,
        )
        .size()
        .reset_index(name="species_n")
    )

    band_summary.insert(
        0,
        "summary_type",
        "sampling_band_by_decision",
    )

    summary_parts.append(
        band_summary[
            [
                "summary_type",
                "sampling_band",
                "match_decision",
                "species_n",
            ]
        ]
    )

    summary = pd.concat(
        summary_parts,
        ignore_index=True,
    )

    summary["percentage_of_200"] = (
        summary["species_n"] / len(pilot) * 100
    )

    atomic_write_csv(summary, summary_output_path)

    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(f"Input pilot species: {len(pilot):,}")
    print(f"Automatically accepted: {len(accepted):,}")
    print(f"Manual review required: {len(review):,}")
    print(f"No match/API failure: {len(failed):,}")

    print("\nMatch decisions:")
    print(
        final["match_decision"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nInput-name flags:")
    print(
        final["input_name_flag"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSaved files:")
    for path in [
        all_output_path,
        accepted_output_path,
        review_output_path,
        failed_output_path,
        summary_output_path,
        checkpoint_path,
    ]:
        print(path)

    unresolved_api = final["api_request_status"].ne("success").sum()

    if unresolved_api:
        print(
            f"\nWARNING: {unresolved_api:,} names still have API failures."
        )
        print(
            "Run the same command again. Successful checkpointed names "
            "will be skipped and failed requests will be retried."
        )
        sys.exit(2)

    print("\nGBIF species-name matching completed successfully.")


if __name__ == "__main__":
    main()
