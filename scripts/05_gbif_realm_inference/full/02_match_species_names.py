#!/usr/bin/env python3
"""Match all full-dataset species-range candidates to GBIF taxa.

This is the scalable full-dataset counterpart of
``../pilot/01_match_species_names.py``. It preserves the pilot's GBIF request and
automatic-acceptance rules, while using an append-only journal plus periodic
checkpoint compaction so that tens of thousands of names can be resumed safely.

The candidate input is produced by
``01_prepare_candidates.py``.
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_INPUT = "gbif_full_prediction/gbif_full_candidate_species.csv"
DEFAULT_OUTPUT_DIR = "gbif_full_prediction/gbif_species_name_matching"
DEFAULT_SEED_CHECKPOINT = (
    "gbif_species_range_pilot/gbif_species_name_match_checkpoint.csv"
)
EXPECTED_INPUT_SPECIES = 39_527

INPUT_COLUMNS = [
    "input_species_names",
    "candidate_records_n",
    "ptp_high_records_n",
    "ptp_medium_records_n",
    "ptp_low_records_n",
    "ptp_insufficient_records_n",
    "no_ptp_candidate_records_n",
    "ptp_auto_assignment_records_n",
]

CANDIDATE_METADATA_COLUMNS = [
    "species",
    "candidate_records_n",
    "ptp_high_records_n",
    "ptp_medium_records_n",
    "ptp_low_records_n",
    "ptp_insufficient_records_n",
    "no_ptp_candidate_records_n",
    "ptp_auto_assignment_records_n",
]

MATCH_COLUMNS = [
    "query_name",
    "input_name_flag",
    "input_name_species_like",
    "api_request_status",
    "api_http_status",
    "api_error",
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
    "match_decision",
    "match_decision_reason",
    "manual_review_required",
]

OUTPUT_COLUMNS = CANDIDATE_METADATA_COLUMNS + MATCH_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match the full species-range candidate list against the "
            "GBIF Species API with scalable checkpoints."
        )
    )
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seed-checkpoint",
        default=DEFAULT_SEED_CHECKPOINT,
        help=(
            "Optional pilot checkpoint whose successful API results are reused. "
            "Use an empty string to disable seeding."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.20,
        help="Delay in seconds after each successful API request.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--minimum-confidence",
        type=int,
        default=95,
        help="Pilot-compatible EXACT-match automatic acceptance threshold.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Compact the append-only journal into the checkpoint every N requests.",
    )
    parser.add_argument(
        "--allow-input-count-change",
        action="store_true",
        help="Allow an input count other than the expected 39,527 species.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete this full run's checkpoint/journal and query all names again.",
    )
    return parser.parse_args()


def normalise_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def screen_input_name(name: str) -> tuple[str, bool]:
    """Pilot-compatible screening of placeholders and unusual names."""
    if not name:
        return "missing_name", False

    uncertain_pattern = re.compile(
        r"(^|\s)(cf|aff|nr)\.?(?=\s|$)", flags=re.IGNORECASE
    )
    placeholder_pattern = re.compile(
        r"(^|\s)(sp|spp|gen|genus|species|unknown|unidentified)"
        r"\.?(?=\s|$)",
        flags=re.IGNORECASE,
    )
    if uncertain_pattern.search(name):
        return "uncertain_identification", False
    if placeholder_pattern.search(name):
        return "placeholder_or_higher_taxon", False

    tokens = name.split()
    if len(tokens) < 2:
        return "not_binomial", False

    genus_ok = bool(
        re.fullmatch(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", tokens[0])
    )
    epithet_ok = bool(
        re.fullmatch(r"[a-z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", tokens[1])
    )
    if not genus_ok or not epithet_ok:
        return "unusual_name_format", False
    if len(tokens) > 2:
        return "additional_name_tokens", False
    return "clean_binomial", True


def safe_text(value: Any) -> str:
    return "" if value is None else str(value)


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
    """Pilot-compatible GBIF request constrained to Animalia/Coleoptera."""
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
                GBIF_MATCH_URL, params=parameters, timeout=timeout
            )
            if response.status_code == 200:
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("GBIF response was not a JSON object.")
                return {
                    "request_status": "success",
                    "http_status": response.status_code,
                    "error": "",
                    "result": result,
                }

            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(2 ** (attempt - 1), 30))
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
                time.sleep(min(2 ** (attempt - 1), 30))

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
    """Apply exactly the pilot's automatic acceptance logic."""
    if api_status != "success":
        return "api_failed", "GBIF API request failed; retry is required.", True

    match_type = safe_text(gbif_result.get("matchType")).upper()
    confidence = safe_int(gbif_result.get("confidence"))
    rank = safe_text(gbif_result.get("rank")).upper()
    status = safe_text(gbif_result.get("status")).upper()
    kingdom = safe_text(gbif_result.get("kingdom")).lower()
    order = safe_text(gbif_result.get("order")).lower()
    usage_key = safe_int(gbif_result.get("usageKey"))
    accepted_usage_key = safe_int(gbif_result.get("acceptedUsageKey"))

    if match_type in {"", "NONE"} or (
        usage_key is None and accepted_usage_key is None
    ):
        return "no_match", "GBIF returned no usable taxon match.", True
    if not input_species_like:
        return "review", f"Input name flag: {input_flag}.", True
    if match_type == "HIGHERRANK":
        return "review", "GBIF matched only a higher taxonomic rank.", True
    if rank != "SPECIES":
        return "review", f"GBIF matched rank is {rank or 'missing'}, not SPECIES.", True
    if kingdom and kingdom != "animalia":
        return "review", f"Matched kingdom is {kingdom}, not Animalia.", True
    if not order:
        return "review", "Matched order is missing and Coleoptera cannot be verified.", True
    if order != "coleoptera":
        return "review", f"Matched order is {order}, not Coleoptera.", True
    if status not in {"ACCEPTED", "SYNONYM"}:
        return "review", f"GBIF taxonomic status is {status or 'missing'}.", True
    if match_type != "EXACT":
        return "review", f"GBIF match type is {match_type}, not EXACT.", True
    if confidence is None:
        return "review", "GBIF confidence is missing.", True
    if confidence < minimum_confidence:
        return (
            "review",
            f"GBIF confidence {confidence} is below the automatic threshold "
            f"{minimum_confidence}.",
            True,
        )
    if status == "SYNONYM":
        return (
            "accepted_synonym",
            "Exact high-confidence species-level Coleoptera match; GBIF treats "
            "the input name as a synonym.",
            False,
        )
    return (
        "accepted_exact",
        "Exact high-confidence accepted species-level Coleoptera match.",
        False,
    )


def build_output_row(
    input_row: dict[str, Any],
    api_response: dict[str, Any],
    minimum_confidence: int,
) -> dict[str, Any]:
    name = normalise_name(input_row.get("input_species_names"))
    input_flag, species_like = screen_input_name(name)
    result = api_response.get("result", {})
    alternatives = result.get("alternatives", [])
    if not isinstance(alternatives, list):
        alternatives = []

    usage_key = safe_int(result.get("usageKey"))
    accepted_usage_key = safe_int(result.get("acceptedUsageKey"))
    occurrence_taxon_key = accepted_usage_key or usage_key
    decision, reason, manual_review = classify_match(
        input_flag,
        species_like,
        api_response.get("request_status", "failed"),
        result,
        minimum_confidence,
    )

    output = {
        "species": name,
        **{col: input_row.get(col) for col in INPUT_COLUMNS[1:]},
        "query_name": name,
        "input_name_flag": input_flag,
        "input_name_species_like": species_like,
        "api_request_status": api_response.get("request_status", "failed"),
        "api_http_status": api_response.get("http_status"),
        "api_error": api_response.get("error", ""),
        "gbif_usageKey": usage_key,
        "gbif_acceptedUsageKey": accepted_usage_key,
        "gbif_occurrence_taxonKey": occurrence_taxon_key,
        "gbif_scientificName": result.get("scientificName"),
        "gbif_acceptedScientificName": result.get("acceptedScientificName"),
        "gbif_canonicalName": result.get("canonicalName"),
        "gbif_rank": result.get("rank"),
        "gbif_status": result.get("status"),
        "gbif_matchType": result.get("matchType"),
        "gbif_confidence": result.get("confidence"),
        "gbif_kingdom": result.get("kingdom"),
        "gbif_phylum": result.get("phylum"),
        "gbif_class": result.get("class"),
        "gbif_order": result.get("order"),
        "gbif_family": result.get("family"),
        "gbif_genus": result.get("genus"),
        "gbif_note": result.get("note"),
        "gbif_alternatives_n": len(alternatives),
        "gbif_alternatives_json": json.dumps(
            alternatives, ensure_ascii=False, separators=(",", ":")
        ),
        "match_decision": decision,
        "match_decision_reason": reason,
        "manual_review_required": manual_review,
    }
    return {column: output.get(column) for column in OUTPUT_COLUMNS}


def read_result_file(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        data = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Pilot checkpoints used "species" and different metadata columns, but all
    # GBIF result fields have the same names.
    if "query_name" not in data.columns and "species" in data.columns:
        data["query_name"] = data["species"].map(normalise_name)
    for column in OUTPUT_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data[OUTPUT_COLUMNS]


def refresh_candidate_metadata(
    results: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    metadata = candidates.rename(columns={"input_species_names": "species"}).copy()
    metadata["species"] = metadata["species"].map(normalise_name)
    lookup = metadata.set_index("species")
    results = results.copy()
    results["query_name"] = results["query_name"].map(normalise_name)
    results["species"] = results["query_name"]
    for column in CANDIDATE_METADATA_COLUMNS[1:]:
        results[column] = results["query_name"].map(lookup[column])
    return results.reindex(columns=OUTPUT_COLUMNS)


def combine_results(
    paths: list[Path], candidates: pd.DataFrame
) -> pd.DataFrame:
    parts = [read_result_file(path) for path in paths if path and path.exists()]
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    combined["query_name"] = combined["query_name"].map(normalise_name)
    combined = combined.loc[combined["query_name"].ne("")]
    combined = combined.drop_duplicates(subset=["query_name"], keep="last")
    valid_names = set(candidates["input_species_names"])
    combined = combined.loc[combined["query_name"].isin(valid_names)].copy()
    return refresh_candidate_metadata(combined, candidates)


def compact_checkpoint(
    checkpoint_path: Path,
    journal_path: Path,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    combined = combine_results([checkpoint_path, journal_path], candidates)
    atomic_write_csv(combined, checkpoint_path)
    if journal_path.exists():
        journal_path.unlink()
    return combined


def initialise_checkpoint(
    checkpoint_path: Path,
    journal_path: Path,
    seed_path: Path | None,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    paths = []
    if seed_path is not None and seed_path.exists():
        paths.append(seed_path)
    paths.extend([checkpoint_path, journal_path])
    combined = combine_results(paths, candidates)
    atomic_write_csv(combined, checkpoint_path)
    if journal_path.exists():
        journal_path.unlink()
    return combined


def append_journal_row(handle: Any, row: dict[str, Any], write_header: bool) -> None:
    writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    writer.writerow({column: row.get(column) for column in OUTPUT_COLUMNS})
    handle.flush()


def write_final_outputs(
    final: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: Path,
) -> int:
    order_lookup = {
        name: position
        for position, name in enumerate(candidates["input_species_names"])
    }
    final = refresh_candidate_metadata(final, candidates)
    final["_input_order"] = final["query_name"].map(order_lookup)
    final = (
        final.sort_values("_input_order")
        .drop(columns="_input_order")
        .reset_index(drop=True)
        .reindex(columns=OUTPUT_COLUMNS)
    )

    accepted = final.loc[
        final["match_decision"].isin(["accepted_exact", "accepted_synonym"])
    ].copy()
    review = final.loc[final["match_decision"].eq("review")].copy()
    failed = final.loc[
        final["match_decision"].isin(["no_match", "api_failed"])
    ].copy()

    paths = {
        "all": output_dir / "gbif_species_name_matches_all.csv",
        "accepted": output_dir / "gbif_species_name_accepted.csv",
        "review": output_dir / "gbif_species_name_review_needed.csv",
        "failed": output_dir / "gbif_species_name_failed.csv",
        "summary": output_dir / "gbif_species_name_match_summary.csv",
    }
    atomic_write_csv(final, paths["all"])
    atomic_write_csv(accepted, paths["accepted"])
    atomic_write_csv(review, paths["review"])
    atomic_write_csv(failed, paths["failed"])

    decision_summary = (
        final.groupby("match_decision", dropna=False)
        .size()
        .reset_index(name="species_n")
    )
    decision_summary.insert(0, "summary_type", "match_decision")
    decision_summary["input_name_flag"] = pd.NA
    flag_summary = (
        final.groupby(["input_name_flag", "match_decision"], dropna=False)
        .size()
        .reset_index(name="species_n")
    )
    flag_summary.insert(0, "summary_type", "input_name_flag_by_decision")
    summary = pd.concat(
        [
            decision_summary[
                ["summary_type", "input_name_flag", "match_decision", "species_n"]
            ],
            flag_summary[
                ["summary_type", "input_name_flag", "match_decision", "species_n"]
            ],
        ],
        ignore_index=True,
    )
    summary["percentage_of_input"] = summary["species_n"] / len(candidates) * 100
    atomic_write_csv(summary, paths["summary"])

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Input species:             {len(candidates):,}")
    print(f"Results in checkpoint:     {len(final):,}")
    print(f"Automatically accepted:    {len(accepted):,}")
    print(f"Manual review required:    {len(review):,}")
    print(f"No match/API failure:      {len(failed):,}")
    print("\nMatch decisions:")
    print(final["match_decision"].value_counts(dropna=False).to_string())
    print("\nSaved files:")
    for path in paths.values():
        print(path)

    unresolved_api = int(final["api_request_status"].ne("success").sum())
    return unresolved_api


def main() -> None:
    args = parse_args()
    if args.sleep < 0 or args.timeout <= 0 or args.max_retries <= 0:
        raise ValueError("sleep, timeout, and max-retries must be valid positive settings")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    seed_path = Path(args.seed_checkpoint) if args.seed_checkpoint else None
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "gbif_species_name_match_checkpoint.csv"
    journal_path = output_dir / "gbif_species_name_match_journal.csv"
    final_paths = [
        output_dir / "gbif_species_name_matches_all.csv",
        output_dir / "gbif_species_name_accepted.csv",
        output_dir / "gbif_species_name_review_needed.csv",
        output_dir / "gbif_species_name_failed.csv",
        output_dir / "gbif_species_name_match_summary.csv",
    ]

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    candidates = pd.read_csv(input_path, low_memory=False)
    missing = sorted(set(INPUT_COLUMNS).difference(candidates.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    candidates = candidates[INPUT_COLUMNS].copy()
    candidates["input_species_names"] = candidates["input_species_names"].map(
        normalise_name
    )
    if candidates["input_species_names"].eq("").any():
        raise ValueError("Input contains missing or empty species names")
    if candidates["input_species_names"].duplicated().any():
        raise ValueError("Input contains duplicated species names")
    if (
        len(candidates) != EXPECTED_INPUT_SPECIES
        and not args.allow_input_count_change
    ):
        raise ValueError(
            f"Expected {EXPECTED_INPUT_SPECIES:,} input species but observed "
            f"{len(candidates):,}. Use --allow-input-count-change only if the "
            "candidate dataset was deliberately changed."
        )

    if args.restart:
        for path in [checkpoint_path, journal_path, *final_paths]:
            if path.exists():
                path.unlink()

    print("=" * 100)
    print("FULL GBIF SPECIES-NAME MATCHING")
    print("=" * 100)
    print(f"Input:                         {input_path}")
    print(f"Input species:                 {len(candidates):,}")
    print(f"Output directory:              {output_dir}")
    print(f"Pilot seed checkpoint:         {seed_path}")
    print(f"Minimum automatic confidence: {args.minimum_confidence}")
    print(f"Checkpoint every:              {args.checkpoint_every:,} requests")
    print(f"Restart requested:             {args.restart}")

    checkpoint = initialise_checkpoint(
        checkpoint_path, journal_path, seed_path, candidates
    )
    completed_names = set(
        checkpoint.loc[
            checkpoint["api_request_status"].eq("success"), "query_name"
        ].dropna().astype(str)
    )
    pending = candidates.loc[
        ~candidates["input_species_names"].isin(completed_names)
    ].copy()
    print(f"Successful names already available: {len(completed_names):,}")
    print(f"Names requiring API request:        {len(pending):,}")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "MSc-Coleoptera-GBIF-Realm-Full/1.0 (academic research)"
            ),
        }
    )

    requests_since_compaction = 0
    processed_this_run = 0
    interrupted = False
    journal_handle = None
    try:
        journal_existed = journal_path.exists() and journal_path.stat().st_size > 0
        journal_needs_header = not journal_existed
        journal_handle = journal_path.open("a", newline="", encoding="utf-8")
        for request_number, input_row in enumerate(
            pending.to_dict(orient="records"), start=1
        ):
            name = normalise_name(input_row["input_species_names"])
            print(
                f"[{request_number:05d}/{len(pending):05d}] {name}",
                end="",
                flush=True,
            )
            api_response = request_gbif_match(
                session, name, args.timeout, args.max_retries
            )
            output_row = build_output_row(
                input_row, api_response, args.minimum_confidence
            )
            append_journal_row(
                journal_handle,
                output_row,
                write_header=journal_needs_header,
            )
            journal_needs_header = False
            processed_this_run += 1
            requests_since_compaction += 1
            print(
                f" -> {output_row['match_decision']} | "
                f"{output_row['gbif_scientificName']} | "
                f"confidence={output_row['gbif_confidence']}"
            )

            if requests_since_compaction >= args.checkpoint_every:
                journal_handle.close()
                journal_handle = None
                checkpoint = compact_checkpoint(
                    checkpoint_path, journal_path, candidates
                )
                print(
                    f"--- checkpoint compacted: {len(checkpoint):,}/"
                    f"{len(candidates):,} species stored ---"
                )
                journal_handle = journal_path.open(
                    "a", newline="", encoding="utf-8"
                )
                journal_needs_header = True
                requests_since_compaction = 0

            if api_response["request_status"] == "success":
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboard interrupt received; preserving checkpoint before exit.")
    finally:
        if journal_handle is not None and not journal_handle.closed:
            journal_handle.close()

    checkpoint = compact_checkpoint(checkpoint_path, journal_path, candidates)
    if interrupted:
        print(
            f"Saved {len(checkpoint):,} species results. Run the same command "
            "again to resume; do not use --restart."
        )
        sys.exit(130)

    unresolved_api = write_final_outputs(checkpoint, candidates, output_dir)
    if unresolved_api:
        print(
            f"\nWARNING: {unresolved_api:,} names still have API failures. "
            "Run the same command again; successful names will be skipped."
        )
        sys.exit(2)
    if len(checkpoint) != len(candidates):
        raise RuntimeError(
            f"Final checkpoint has {len(checkpoint):,} rows, expected "
            f"{len(candidates):,}"
        )
    print("\nFull GBIF species-name matching completed successfully.")


if __name__ == "__main__":
    main()
