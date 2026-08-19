#!/usr/bin/env python3
"""
Finalise the GBIF species-name matching decisions for the 200-species pilot.

The script:
1. reads the automatically accepted and manual-review CSV files;
2. re-runs GBIF Species Match for three manually corrected names;
3. creates an auditable decision table for all review-needed names;
4. combines automatic and manual acceptances;
5. creates a taxonKey-deduplicated table for GBIF occurrence queries.

It never modifies the two input CSV files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"

# These names need to be submitted to GBIF again using a corrected/current
# combination. Acceptance is still conditional on a species-level GBIF result.
CORRECTED_NAMES = {
    "Rhantus cicurus": {
        "corrected_name": "Rhantus cicurius",
        "correction_reason": (
            "Corrected a likely one-letter spelling error from 'cicurus' "
            "to 'cicurius'."
        ),
    },
    "Pseudomordellina nanuloides": {
        "corrected_name": "Mordellistena nanuloides",
        "correction_reason": (
            "Re-matched under the species combination used by the GBIF "
            "backbone, Mordellistena nanuloides."
        ),
    },
    "Centorus elongatus": {
        "corrected_name": "Belopus elongatus",
        "correction_reason": (
            "Re-matched under the current/accepted combination "
            "Belopus elongatus."
        ),
    },
}

# Decisions that do not require another API call.
FIXED_DECISIONS = {
    "Pimelia claudia": {
        "manual_decision": "pending_external_taxonomy",
        "manual_corrected_name": "Pimelia claudia",
        "include_in_range_query": False,
        "manual_review_reason": (
            "The name may represent a real species, but the current GBIF "
            "result is genus-level. A genus taxonKey must not be used as a "
            "species range."
        ),
        "manual_evidence_source": (
            "Original GBIF Species Match result; external species-level "
            "taxonomic evidence should be checked separately."
        ),
    },
    "Paropsis tasmanica": {
        "manual_decision": "pending_external_taxonomy",
        "manual_corrected_name": "Paropsis tasmanica",
        "include_in_range_query": False,
        "manual_review_reason": (
            "The name may represent a real species, but the current GBIF "
            "result is genus-level. A genus taxonKey must not be used as a "
            "species range."
        ),
        "manual_evidence_source": (
            "Original GBIF Species Match result; external species-level "
            "taxonomic evidence should be checked separately."
        ),
    },
    "Clerinae gen": {
        "manual_decision": "excluded_placeholder",
        "manual_corrected_name": "",
        "include_in_range_query": False,
        "manual_review_reason": (
            "Placeholder/higher-taxon label rather than a species name."
        ),
        "manual_evidence_source": (
            "Input-name structure and original GBIF higher-rank match."
        ),
    },
    "Gnaptorina miroshnikovi": {
        "manual_decision": "pending_external_taxonomy",
        "manual_corrected_name": "Gnaptorina miroshnikovi",
        "include_in_range_query": False,
        "manual_review_reason": (
            "The name may represent a real species, but the current GBIF "
            "result is genus-level. A genus taxonKey must not be used as a "
            "species range."
        ),
        "manual_evidence_source": (
            "Original GBIF Species Match result; external species-level "
            "taxonomic evidence should be checked separately."
        ),
    },
    "Chalcodrya variegata": {
        "manual_decision": "pending_external_taxonomy",
        "manual_corrected_name": "Chalcodrya variegata",
        "include_in_range_query": False,
        "manual_review_reason": (
            "The name may represent a real species, but the current GBIF "
            "result is genus-level. A genus taxonKey must not be used as a "
            "species range."
        ),
        "manual_evidence_source": (
            "Original GBIF Species Match result; external species-level "
            "taxonomic evidence should be checked separately."
        ),
    },
    "Hyperaspis signata": {
        "manual_decision": "accepted_manual_doubtful",
        "manual_corrected_name": "Hyperaspis signata",
        "include_in_range_query": True,
        "manual_review_reason": (
            "Exact species-level match retained for the pilot, while "
            "preserving GBIF's DOUBTFUL taxonomic-status warning."
        ),
        "manual_evidence_source": (
            "Original exact GBIF Species Match plus manual taxonomic review."
        ),
    },
    "Platycrepidius duodecimnotatus": {
        "manual_decision": "pending_external_taxonomy",
        "manual_corrected_name": "Platycrepidius duodecimnotatus",
        "include_in_range_query": False,
        "manual_review_reason": (
            "The name may represent a real species, but the current GBIF "
            "result is genus-level. A genus taxonKey must not be used as a "
            "species range."
        ),
        "manual_evidence_source": (
            "Original GBIF Species Match result; external species-level "
            "taxonomic evidence should be checked separately."
        ),
    },
}

GBIF_FIELDS = [
    "scientificName",
    "acceptedScientificName",
    "canonicalName",
    "rank",
    "status",
    "matchType",
    "confidence",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "usageKey",
    "acceptedUsageKey",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalise manual GBIF name decisions and create the safe, "
            "species-level occurrence-query table for the pilot."
        )
    )
    parser.add_argument(
        "--accepted",
        default=(
            "gbif_species_range_pilot/"
            "gbif_species_name_accepted.csv"
        ),
        help="Automatically accepted GBIF name-match CSV.",
    )
    parser.add_argument(
        "--review",
        default=(
            "gbif_species_range_pilot/"
            "gbif_species_name_review_needed.csv"
        ),
        help="GBIF name-match CSV requiring manual review.",
    )
    parser.add_argument(
        "--output-dir",
        default="gbif_species_range_pilot",
        help="Directory for all output files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each GBIF request (default: 30).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Maximum GBIF request attempts per corrected name (default: 4).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay in seconds between successful GBIF requests (default: 0.25).",
    )
    return parser.parse_args()


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def clean_key(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def gbif_species_match(
    name: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    params = urlencode(
        {
            "name": name,
            "kingdom": "Animalia",
            "order": "Coleoptera",
            "verbose": "true",
        }
    )
    url = f"{GBIF_MATCH_URL}?{params}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "MSc-GBIF-species-range-pilot/1.0 "
                "(manual name-resolution audit)"
            ),
        },
    )

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            payload["_api_success"] = True
            payload["_api_error"] = ""
            return payload
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            # Retry rate limits and server errors, but not other client errors.
            if exc.code < 500 and exc.code != 429:
                break
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))

    return {
        "_api_success": False,
        "_api_error": last_error or "Unknown GBIF API error",
    }


def flatten_corrected_match(
    original_name: str,
    corrected_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "species": original_name,
        "corrected_name_submitted": corrected_name,
        "api_success": bool(payload.get("_api_success", False)),
        "api_error": payload.get("_api_error", ""),
        "alternatives_n": len(payload.get("alternatives", []) or []),
    }
    for field in GBIF_FIELDS:
        row[f"corrected_gbif_{field}"] = payload.get(field)

    accepted_key = clean_key(payload.get("acceptedUsageKey"))
    usage_key = clean_key(payload.get("usageKey"))
    row["corrected_gbif_occurrence_taxonKey"] = accepted_key or usage_key
    return row


def corrected_match_is_safe(row: pd.Series) -> tuple[bool, str]:
    problems: list[str] = []

    if not bool(row.get("api_success", False)):
        problems.append(f"API request failed: {clean_text(row.get('api_error'))}")

    if clean_text(row.get("corrected_gbif_rank")).upper() != "SPECIES":
        problems.append("GBIF result is not species-level")

    if clean_text(row.get("corrected_gbif_order")).upper() != "COLEOPTERA":
        problems.append("GBIF result is not in Coleoptera")

    if clean_text(row.get("corrected_gbif_matchType")).upper() != "EXACT":
        problems.append("corrected name did not receive an EXACT match")

    status = clean_text(row.get("corrected_gbif_status")).upper()
    if status not in {"ACCEPTED", "SYNONYM"}:
        problems.append(f"taxonomic status is {status or 'missing'}")

    if clean_key(row.get("corrected_gbif_occurrence_taxonKey")) is None:
        problems.append("no usable occurrence taxonKey")

    return not problems, "; ".join(problems)


def resolved_name_from_row(row: pd.Series, prefix: str = "gbif_") -> str:
    candidates = [
        row.get(f"{prefix}acceptedScientificName"),
        row.get(f"{prefix}scientificName"),
        row.get(f"{prefix}canonicalName"),
        row.get("species"),
    ]
    for candidate in candidates:
        text = clean_text(candidate)
        if text:
            return text
    return ""


def build_manual_decisions(
    review: pd.DataFrame,
    corrected_matches: pd.DataFrame,
) -> pd.DataFrame:
    corrected_lookup = corrected_matches.set_index("species", drop=False)
    rows: list[dict[str, Any]] = []
    today = date.today().isoformat()

    for _, source_row in review.iterrows():
        species = clean_text(source_row["species"])
        decision = source_row.to_dict()
        decision.update(
            {
                "manual_decision": "",
                "manual_corrected_name": "",
                "manual_occurrence_taxonKey": pd.NA,
                "manual_resolved_scientific_name": "",
                "manual_gbif_rank": "",
                "manual_gbif_status": "",
                "manual_gbif_matchType": "",
                "manual_review_reason": "",
                "manual_evidence_source": "",
                "include_in_range_query": False,
                "manual_reviewed_date": today,
            }
        )

        if species in CORRECTED_NAMES:
            match = corrected_lookup.loc[species]
            safe, problem_text = corrected_match_is_safe(match)
            corrected_name = CORRECTED_NAMES[species]["corrected_name"]
            correction_reason = CORRECTED_NAMES[species]["correction_reason"]

            decision["manual_corrected_name"] = corrected_name
            decision["manual_evidence_source"] = (
                "GBIF Species Match API re-run on the corrected name."
            )

            if safe:
                decision["manual_decision"] = "accepted_manual_correction"
                decision["manual_occurrence_taxonKey"] = clean_key(
                    match["corrected_gbif_occurrence_taxonKey"]
                )
                decision["manual_resolved_scientific_name"] = (
                    resolved_name_from_row(match, prefix="corrected_gbif_")
                )
                decision["manual_gbif_rank"] = clean_text(
                    match.get("corrected_gbif_rank")
                )
                decision["manual_gbif_status"] = clean_text(
                    match.get("corrected_gbif_status")
                )
                decision["manual_gbif_matchType"] = clean_text(
                    match.get("corrected_gbif_matchType")
                )
                decision["include_in_range_query"] = True
                decision["manual_review_reason"] = (
                    f"{correction_reason} The corrected name received a safe "
                    "species-level EXACT GBIF match."
                )
            else:
                decision["manual_decision"] = (
                    "pending_corrected_name_verification"
                )
                decision["include_in_range_query"] = False
                decision["manual_review_reason"] = (
                    f"{correction_reason} The corrected result was not "
                    f"automatically accepted because: {problem_text}."
                )

        elif species in FIXED_DECISIONS:
            fixed = FIXED_DECISIONS[species]
            decision.update(fixed)

            if fixed["include_in_range_query"]:
                original_rank = clean_text(
                    source_row.get("gbif_rank")
                ).upper()
                original_order = clean_text(
                    source_row.get("gbif_order")
                ).upper()
                original_key = clean_key(
                    source_row.get("gbif_occurrence_taxonKey")
                )
                if (
                    original_rank == "SPECIES"
                    and original_order == "COLEOPTERA"
                    and original_key is not None
                ):
                    decision["manual_occurrence_taxonKey"] = original_key
                    decision["manual_resolved_scientific_name"] = (
                        resolved_name_from_row(source_row)
                    )
                    decision["manual_gbif_rank"] = clean_text(
                        source_row.get("gbif_rank")
                    )
                    decision["manual_gbif_status"] = clean_text(
                        source_row.get("gbif_status")
                    )
                    decision["manual_gbif_matchType"] = clean_text(
                        source_row.get("gbif_matchType")
                    )
                else:
                    decision["manual_decision"] = (
                        "pending_manual_key_verification"
                    )
                    decision["include_in_range_query"] = False
                    decision["manual_review_reason"] += (
                        " However, the stored match failed the final "
                        "species/order/taxonKey safety check."
                    )
        else:
            # This makes the script safe if the review file later contains a
            # name not covered by the current 10-name pilot decisions.
            decision["manual_decision"] = "pending_manual_review"
            decision["manual_review_reason"] = (
                "No pre-defined manual decision exists for this name."
            )
            decision["manual_evidence_source"] = "Not yet reviewed."

        rows.append(decision)

    manual = pd.DataFrame(rows)
    manual["manual_occurrence_taxonKey"] = pd.array(
        manual["manual_occurrence_taxonKey"], dtype="Int64"
    )
    manual["include_in_range_query"] = (
        manual["include_in_range_query"].fillna(False).astype(bool)
    )
    return manual


def validate_automatic_acceptances(accepted: pd.DataFrame) -> None:
    rank_ok = (
        accepted["gbif_rank"].fillna("").astype(str).str.upper() == "SPECIES"
    )
    order_ok = (
        accepted["gbif_order"].fillna("").astype(str).str.upper()
        == "COLEOPTERA"
    )
    key_ok = accepted["gbif_occurrence_taxonKey"].map(clean_key).notna()
    valid = rank_ok & order_ok & key_ok

    if not valid.all():
        bad_names = accepted.loc[~valid, "species"].astype(str).tolist()
        preview = ", ".join(bad_names[:10])
        raise ValueError(
            "The automatically accepted file contains rows that fail the "
            "species-level/order/taxonKey safety checks. Refusing to create "
            f"the query table. Example name(s): {preview}"
        )


def build_final_accepted_names(
    accepted: pd.DataFrame,
    manual: pd.DataFrame,
) -> pd.DataFrame:
    automatic_rows: list[dict[str, Any]] = []
    for _, row in accepted.iterrows():
        automatic_rows.append(
            {
                "species": clean_text(row["species"]),
                "resolved_scientific_name": resolved_name_from_row(row),
                "match_decision": clean_text(row["match_decision"]),
                "decision_source": "automatic",
                "gbif_status": clean_text(row.get("gbif_status")),
                "gbif_occurrence_taxonKey": clean_key(
                    row.get("gbif_occurrence_taxonKey")
                ),
                "sampling_band": clean_text(row.get("sampling_band")),
                "missing_records_n": pd.to_numeric(
                    row.get("missing_records_n"), errors="coerce"
                ),
            }
        )

    manual_rows: list[dict[str, Any]] = []
    included_manual = manual[manual["include_in_range_query"]].copy()
    for _, row in included_manual.iterrows():
        manual_rows.append(
            {
                "species": clean_text(row["species"]),
                "resolved_scientific_name": clean_text(
                    row.get("manual_resolved_scientific_name")
                )
                or clean_text(row.get("manual_corrected_name")),
                "match_decision": clean_text(row["manual_decision"]),
                "decision_source": "manual",
                "gbif_status": (
                    clean_text(row.get("manual_gbif_status"))
                    or clean_text(row.get("gbif_status"))
                ),
                "gbif_occurrence_taxonKey": clean_key(
                    row.get("manual_occurrence_taxonKey")
                ),
                "sampling_band": clean_text(row.get("sampling_band")),
                "missing_records_n": pd.to_numeric(
                    row.get("missing_records_n"), errors="coerce"
                ),
            }
        )

    final = pd.DataFrame(automatic_rows + manual_rows)
    final["gbif_occurrence_taxonKey"] = pd.array(
        final["gbif_occurrence_taxonKey"], dtype="Int64"
    )
    final["missing_records_n"] = (
        pd.to_numeric(final["missing_records_n"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )

    if final["gbif_occurrence_taxonKey"].isna().any():
        raise ValueError(
            "An accepted final row has no occurrence taxonKey. "
            "No outputs were finalised."
        )

    return final.sort_values(
        ["gbif_occurrence_taxonKey", "species"]
    ).reset_index(drop=True)


def join_unique(series: pd.Series) -> str:
    values = sorted(
        {
            clean_text(value)
            for value in series
            if clean_text(value)
        }
    )
    return " | ".join(values)


def build_occurrence_query_table(
    final_names: pd.DataFrame,
) -> pd.DataFrame:
    # One request per accepted GBIF taxonKey. This avoids downloading the same
    # occurrence range more than once when several input synonyms converge.
    query = (
        final_names.groupby(
            "gbif_occurrence_taxonKey",
            as_index=False,
            dropna=False,
        )
        .agg(
            resolved_scientific_name=(
                "resolved_scientific_name",
                join_unique,
            ),
            input_species_names=("species", join_unique),
            input_species_n=("species", "nunique"),
            sampling_bands=("sampling_band", join_unique),
            missing_records_n=("missing_records_n", "sum"),
            match_decisions=("match_decision", join_unique),
            decision_sources=("decision_source", join_unique),
            gbif_statuses=("gbif_status", join_unique),
        )
        .sort_values("gbif_occurrence_taxonKey")
        .reset_index(drop=True)
    )
    query["gbif_occurrence_taxonKey"] = pd.array(
        query["gbif_occurrence_taxonKey"], dtype="Int64"
    )
    query["input_species_n"] = query["input_species_n"].astype("Int64")
    query["missing_records_n"] = query["missing_records_n"].astype("Int64")
    return query


def build_summary(
    accepted: pd.DataFrame,
    manual: pd.DataFrame,
    final_names: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ("automatic_accepted_input_names", len(accepted)),
        ("manual_review_input_names", len(manual)),
        (
            "manual_names_included",
            int(manual["include_in_range_query"].sum()),
        ),
        (
            "manual_names_not_included",
            int((~manual["include_in_range_query"]).sum()),
        ),
        ("final_accepted_input_names", len(final_names)),
        ("unique_occurrence_taxonKeys", len(query)),
        (
            "duplicate_input_names_collapsed_by_taxonKey",
            len(final_names) - len(query),
        ),
    ]
    for decision, count in (
        manual["manual_decision"].value_counts(dropna=False).items()
    ):
        rows.append((f"manual_decision::{decision}", int(count)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> int:
    args = parse_args()
    accepted_path = Path(args.accepted)
    review_path = Path(args.review)
    output_dir = Path(args.output_dir)

    if args.retries < 1:
        raise ValueError("--retries must be at least 1.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if not accepted_path.is_file():
        raise FileNotFoundError(f"Accepted file not found: {accepted_path}")
    if not review_path.is_file():
        raise FileNotFoundError(f"Review file not found: {review_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("LOADING PILOT NAME-MATCH FILES")
    print("=" * 88)
    accepted = pd.read_csv(accepted_path, low_memory=False)
    review = pd.read_csv(review_path, low_memory=False)

    require_columns(
        accepted,
        [
            "species",
            "match_decision",
            "gbif_rank",
            "gbif_order",
            "gbif_occurrence_taxonKey",
        ],
        "Automatically accepted file",
    )
    require_columns(
        review,
        [
            "species",
            "gbif_rank",
            "gbif_order",
            "gbif_status",
            "gbif_occurrence_taxonKey",
        ],
        "Manual-review file",
    )

    duplicate_review = review["species"].astype(str).duplicated(keep=False)
    if duplicate_review.any():
        names = sorted(review.loc[duplicate_review, "species"].astype(str).unique())
        raise ValueError(
            "The manual-review file contains duplicate species names: "
            + ", ".join(names)
        )

    expected = set(CORRECTED_NAMES) | set(FIXED_DECISIONS)
    observed = set(review["species"].dropna().astype(str).str.strip())
    missing_expected = sorted(expected - observed)
    extra_observed = sorted(observed - expected)

    print(f"Automatically accepted names: {len(accepted):,}")
    print(f"Names requiring review:       {len(review):,}")
    if missing_expected:
        print(
            "Warning: expected pilot review name(s) not present: "
            + ", ".join(missing_expected)
        )
    if extra_observed:
        print(
            "Warning: additional name(s) will remain pending manual review: "
            + ", ".join(extra_observed)
        )

    validate_automatic_acceptances(accepted)

    print("\n" + "=" * 88)
    print("RE-MATCHING THREE CORRECTED NAMES WITH GBIF")
    print("=" * 88)
    corrected_rows: list[dict[str, Any]] = []
    for index, (original_name, metadata) in enumerate(
        CORRECTED_NAMES.items(), start=1
    ):
        corrected_name = metadata["corrected_name"]
        print(
            f"[{index}/{len(CORRECTED_NAMES)}] "
            f"{original_name} -> {corrected_name}"
        )
        payload = gbif_species_match(
            corrected_name,
            timeout=args.timeout,
            retries=args.retries,
        )
        corrected_rows.append(
            flatten_corrected_match(
                original_name,
                corrected_name,
                payload,
            )
        )
        if payload.get("_api_success") and index < len(CORRECTED_NAMES):
            time.sleep(args.delay)

    corrected_matches = pd.DataFrame(corrected_rows)
    corrected_matches["corrected_gbif_occurrence_taxonKey"] = pd.array(
        corrected_matches["corrected_gbif_occurrence_taxonKey"],
        dtype="Int64",
    )

    corrected_path = output_dir / "gbif_corrected_name_matches.csv"
    corrected_matches.to_csv(corrected_path, index=False)

    print("\n" + "=" * 88)
    print("BUILDING MANUAL DECISIONS AND FINAL QUERY TABLE")
    print("=" * 88)
    manual = build_manual_decisions(review, corrected_matches)
    final_names = build_final_accepted_names(accepted, manual)
    query = build_occurrence_query_table(final_names)
    summary = build_summary(
        accepted,
        manual,
        final_names,
        query,
    )

    manual_path = output_dir / "gbif_species_name_manual_decisions.csv"
    final_names_path = (
        output_dir / "gbif_species_name_final_accepted_names.csv"
    )
    query_path = (
        output_dir / "gbif_species_name_final_for_occurrence.csv"
    )
    summary_path = (
        output_dir / "gbif_species_name_finalisation_summary.csv"
    )

    manual.to_csv(manual_path, index=False)
    final_names.to_csv(final_names_path, index=False)
    query.to_csv(query_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Manual review decisions:      {len(manual):,}")
    print(
        "Manual names included:       "
        f"{int(manual['include_in_range_query'].sum()):,}"
    )
    print(f"Final accepted input names:   {len(final_names):,}")
    print(f"Unique occurrence taxonKeys:  {len(query):,}")

    print("\nManual decisions:")
    print(manual["manual_decision"].value_counts(dropna=False).to_string())

    print("\nSaved files:")
    for path in [
        corrected_path,
        manual_path,
        final_names_path,
        query_path,
        summary_path,
    ]:
        print(path)

    print("\nGBIF pilot name finalisation completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
