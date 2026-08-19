#!/usr/bin/env python3
"""
Download and quality-filter GBIF occurrences for the species-range pilot.

This script is designed for a feasibility pilot, not as a replacement for a
formal GBIF occurrence download with a DOI.

Input
-----
gbif_species_range_pilot/gbif_species_name_final_for_occurrence.csv

Main outputs
------------
gbif_occurrence_records_for_realm.csv
    Quality-filtered, coordinate-deduplicated and spatially balanced occurrence
    records to use in the next realm-overlay step.

gbif_occurrence_species_summary.csv
    Per-taxon coverage and quality-control statistics.

gbif_occurrence_pilot_summary.csv
    Overall pilot statistics.

gbif_occurrence_failed_taxa.csv
    Taxa that could not be completed in this run.

The script keeps one checkpoint CSV and one metadata JSON per taxonKey, so an
interrupted run can be resumed safely with the same command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


GBIF_OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_PAGE_LIMIT = 300

DEFAULT_INPUT = (
    "gbif_species_range_pilot/"
    "gbif_species_name_final_for_occurrence.csv"
)
DEFAULT_OUTPUT_DIR = (
    "gbif_species_range_pilot/"
    "gbif_occurrence_pilot"
)

# Fossils do not represent the present-day range. Living specimens may refer
# to captive or collection holdings rather than wild occurrence localities.
EXCLUDED_BASIS_OF_RECORD = {
    "FOSSIL_SPECIMEN",
    "LIVING_SPECIMEN",
}

OCCURRENCE_COLUMNS = [
    "gbifID",
    "key",
    "scientificName",
    "acceptedScientificName",
    "taxonKey",
    "acceptedTaxonKey",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "coordinatePrecision",
    "geodeticDatum",
    "countryCode",
    "country",
    "stateProvince",
    "locality",
    "year",
    "month",
    "day",
    "eventDate",
    "basisOfRecord",
    "occurrenceStatus",
    "establishmentMeans",
    "degreeOfEstablishment",
    "typeStatus",
    "datasetKey",
    "publishingOrgKey",
    "institutionCode",
    "collectionCode",
    "catalogNumber",
    "recordedBy",
    "identifiedBy",
    "issues",
    "license",
]

OUTPUT_RECORD_COLUMNS = OCCURRENCE_COLUMNS + [
    "query_taxonKey",
    "query_scientific_name",
    "input_species_names",
    "input_missing_records_n",
    "input_sampling_bands",
    "input_match_decisions",
    "input_gbif_statuses",
    "taxonomic_caution",
    "sampling_grid_degrees",
    "sampling_grid_cell",
    "spatial_sampling_applied",
]

FAILED_COLUMNS = [
    "gbif_occurrence_taxonKey",
    "resolved_scientific_name",
    "input_species_names",
    "error",
    "failed_at_utc",
]

INPUT_REQUIRED_COLUMNS = [
    "gbif_occurrence_taxonKey",
    "resolved_scientific_name",
    "input_species_names",
    "missing_records_n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download, quality-filter and spatially balance GBIF occurrence "
            "records for the species-range feasibility pilot."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Final, taxonKey-deduplicated species query table.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory, including per-taxon checkpoints.",
    )
    parser.add_argument(
        "--max-fetch-per-species",
        type=int,
        default=5000,
        help=(
            "Maximum API records retrieved per taxonKey after GBIF server-side "
            "filters (default: 5000)."
        ),
    )
    parser.add_argument(
        "--max-final-per-species",
        type=int,
        default=1000,
        help=(
            "Maximum locally filtered records retained per taxonKey "
            "(default: 1000)."
        ),
    )
    parser.add_argument(
        "--max-coordinate-uncertainty-m",
        type=float,
        default=100000.0,
        help=(
            "Exclude records with a known uncertainty above this threshold; "
            "records with missing uncertainty are retained (default: 100000)."
        ),
    )
    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=5,
        help=(
            "Decimal places used to identify duplicate coordinates "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--balance-grid-degrees",
        type=float,
        default=1.0,
        help=(
            "Grid-cell size used for spatially balanced subsampling "
            "(default: 1 degree)."
        ),
    )
    parser.add_argument(
        "--max-taxa",
        type=int,
        default=None,
        help=(
            "Process only the first N unfinished taxa in this run. Useful for "
            "a small test; omit to process all unfinished taxa."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds per GBIF request (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum attempts per GBIF page (default: 5).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.15,
        help="Delay in seconds between successful API pages (default: 0.15).",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def clean_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_columns(
    data: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    data.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def request_json(
    params: dict[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = f"{GBIF_OCCURRENCE_SEARCH_URL}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "MSc-GBIF-species-range-pilot/1.0 "
                "(occurrence feasibility analysis)"
            ),
        },
    )

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code < 500 and exc.code != 429:
                break
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 16))

    raise RuntimeError(last_error or "Unknown GBIF API error")


def base_query_params(taxon_key: int) -> dict[str, Any]:
    return {
        "taxon_key": taxon_key,
        "has_coordinate": "true",
        "has_geospatial_issue": "false",
        "occurrence_status": "PRESENT",
    }


def flatten_occurrence(
    record: dict[str, Any],
    query_row: pd.Series,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for column in OCCURRENCE_COLUMNS:
        value = record.get(column)
        if column == "issues" and isinstance(value, list):
            value = " | ".join(str(item) for item in value)
        elif isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        flattened[column] = value

    flattened["query_taxonKey"] = clean_int(
        query_row["gbif_occurrence_taxonKey"]
    )
    flattened["query_scientific_name"] = clean_text(
        query_row["resolved_scientific_name"]
    )
    flattened["input_species_names"] = clean_text(
        query_row["input_species_names"]
    )
    flattened["input_missing_records_n"] = clean_int(
        query_row["missing_records_n"]
    )
    flattened["input_sampling_bands"] = clean_text(
        query_row.get("sampling_bands")
    )
    flattened["input_match_decisions"] = clean_text(
        query_row.get("match_decisions")
    )
    statuses = clean_text(query_row.get("gbif_statuses"))
    flattened["input_gbif_statuses"] = statuses
    flattened["taxonomic_caution"] = (
        "GBIF_DOUBTFUL" if "DOUBTFUL" in statuses.upper() else ""
    )
    return flattened


def download_taxon_occurrences(
    query_row: pd.Series,
    max_fetch: int,
    timeout: float,
    retries: int,
    request_delay: float,
) -> tuple[pd.DataFrame, int]:
    taxon_key = clean_int(query_row["gbif_occurrence_taxonKey"])
    if taxon_key is None:
        raise ValueError("Missing or invalid gbif_occurrence_taxonKey")

    records: list[dict[str, Any]] = []
    offset = 0
    api_total: int | None = None

    while len(records) < max_fetch:
        page_limit = min(
            GBIF_PAGE_LIMIT,
            max_fetch - len(records),
        )
        params = base_query_params(taxon_key)
        params.update({"limit": page_limit, "offset": offset})
        payload = request_json(params, timeout=timeout, retries=retries)

        if api_total is None:
            api_total = clean_int(payload.get("count")) or 0

        page = payload.get("results", []) or []
        for record in page:
            records.append(flatten_occurrence(record, query_row))

        if (
            not page
            or bool(payload.get("endOfRecords", False))
            or len(records) >= api_total
        ):
            break

        offset += len(page)
        time.sleep(request_delay)

    return pd.DataFrame(records), int(api_total or 0)


def deterministic_hash(row: pd.Series, taxon_key: int) -> str:
    identity = "|".join(
        [
            str(taxon_key),
            clean_text(row.get("gbifID")),
            clean_text(row.get("key")),
            clean_text(row.get("decimalLatitude")),
            clean_text(row.get("decimalLongitude")),
            clean_text(row.get("datasetKey")),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def spatially_balanced_sample(
    unique_coordinates: pd.DataFrame,
    taxon_key: int,
    maximum: int,
    grid_degrees: float,
) -> pd.DataFrame:
    if unique_coordinates.empty:
        return unique_coordinates.copy()

    data = unique_coordinates.copy()
    data["_stable_hash"] = data.apply(
        deterministic_hash,
        axis=1,
        taxon_key=taxon_key,
    )
    data["_grid_lat"] = (
        (data["decimalLatitude"] + 90.0) / grid_degrees
    ).map(math.floor)
    data["_grid_lon"] = (
        (data["decimalLongitude"] + 180.0) / grid_degrees
    ).map(math.floor)
    data["_grid_id"] = (
        data["_grid_lat"].astype(str)
        + "_"
        + data["_grid_lon"].astype(str)
    )
    data = data.sort_values("_stable_hash").reset_index(drop=True)

    # First retain one deterministic record from every occupied grid cell.
    first_per_cell = data.drop_duplicates("_grid_id", keep="first")

    if len(first_per_cell) >= maximum:
        selected = first_per_cell.head(maximum).copy()
    else:
        selected_indices = set(first_per_cell.index)
        remaining = data.loc[~data.index.isin(selected_indices)]
        spaces = maximum - len(first_per_cell)
        selected = pd.concat(
            [first_per_cell, remaining.head(spaces)],
            ignore_index=False,
        )

    selected = selected.sort_values("_stable_hash").head(maximum).copy()
    selected["sampling_grid_degrees"] = grid_degrees
    selected["sampling_grid_cell"] = selected["_grid_id"]
    selected["spatial_sampling_applied"] = len(data) > maximum
    return selected.drop(
        columns=["_stable_hash", "_grid_lat", "_grid_lon", "_grid_id"],
        errors="ignore",
    ).reset_index(drop=True)


def quality_filter_and_sample(
    raw: pd.DataFrame,
    taxon_key: int,
    max_uncertainty_m: float,
    coordinate_decimals: int,
    max_final: int,
    grid_degrees: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "records_fetched": len(raw),
        "removed_invalid_coordinates": 0,
        "removed_zero_zero": 0,
        "removed_high_uncertainty": 0,
        "removed_excluded_basis": 0,
        "records_after_local_qc": 0,
        "removed_duplicate_coordinates": 0,
        "unique_coordinate_records": 0,
        "final_sample_records": 0,
    }
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_RECORD_COLUMNS), stats

    data = raw.copy()
    data["decimalLatitude"] = pd.to_numeric(
        data["decimalLatitude"], errors="coerce"
    )
    data["decimalLongitude"] = pd.to_numeric(
        data["decimalLongitude"], errors="coerce"
    )
    data["coordinateUncertaintyInMeters"] = pd.to_numeric(
        data["coordinateUncertaintyInMeters"], errors="coerce"
    )

    valid_coordinates = (
        data["decimalLatitude"].between(-90, 90, inclusive="both")
        & data["decimalLongitude"].between(-180, 180, inclusive="both")
    )
    stats["removed_invalid_coordinates"] = int((~valid_coordinates).sum())
    data = data.loc[valid_coordinates].copy()

    zero_zero = (
        data["decimalLatitude"].eq(0)
        & data["decimalLongitude"].eq(0)
    )
    stats["removed_zero_zero"] = int(zero_zero.sum())
    data = data.loc[~zero_zero].copy()

    high_uncertainty = (
        data["coordinateUncertaintyInMeters"].notna()
        & (
            data["coordinateUncertaintyInMeters"]
            > max_uncertainty_m
        )
    )
    stats["removed_high_uncertainty"] = int(high_uncertainty.sum())
    data = data.loc[~high_uncertainty].copy()

    basis = data["basisOfRecord"].fillna("").astype(str).str.upper()
    excluded_basis = basis.isin(EXCLUDED_BASIS_OF_RECORD)
    stats["removed_excluded_basis"] = int(excluded_basis.sum())
    data = data.loc[~excluded_basis].copy()
    stats["records_after_local_qc"] = len(data)

    data["_lat_dedup"] = data["decimalLatitude"].round(
        coordinate_decimals
    )
    data["_lon_dedup"] = data["decimalLongitude"].round(
        coordinate_decimals
    )
    data["_dedup_hash"] = data.apply(
        deterministic_hash,
        axis=1,
        taxon_key=taxon_key,
    )
    data = data.sort_values("_dedup_hash")
    before_dedup = len(data)
    data = data.drop_duplicates(
        subset=["_lat_dedup", "_lon_dedup"],
        keep="first",
    ).copy()
    stats["removed_duplicate_coordinates"] = before_dedup - len(data)
    stats["unique_coordinate_records"] = len(data)
    data = data.drop(
        columns=["_lat_dedup", "_lon_dedup", "_dedup_hash"],
        errors="ignore",
    )

    final = spatially_balanced_sample(
        data,
        taxon_key=taxon_key,
        maximum=max_final,
        grid_degrees=grid_degrees,
    )
    stats["final_sample_records"] = len(final)
    return final, stats


def config_signature(
    query_row: pd.Series,
    args: argparse.Namespace,
) -> str:
    payload = {
        "taxonKey": clean_int(query_row["gbif_occurrence_taxonKey"]),
        "resolved_scientific_name": clean_text(
            query_row["resolved_scientific_name"]
        ),
        "input_species_names": clean_text(query_row["input_species_names"]),
        "max_fetch_per_species": args.max_fetch_per_species,
        "max_final_per_species": args.max_final_per_species,
        "max_coordinate_uncertainty_m": (
            args.max_coordinate_uncertainty_m
        ),
        "coordinate_decimals": args.coordinate_decimals,
        "balance_grid_degrees": args.balance_grid_degrees,
        "server_filters": base_query_params(
            clean_int(query_row["gbif_occurrence_taxonKey"]) or -1
        ),
        "excluded_basis_of_record": sorted(EXCLUDED_BASIS_OF_RECORD),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_paths(
    checkpoint_dir: Path,
    taxon_key: int,
) -> tuple[Path, Path]:
    return (
        checkpoint_dir / f"{taxon_key}.csv",
        checkpoint_dir / f"{taxon_key}.json",
    )


def load_valid_checkpoint(
    csv_path: Path,
    json_path: Path,
    expected_signature: str,
) -> dict[str, Any] | None:
    if not csv_path.exists() or not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    existing_signature = metadata.get("config_signature")
    if existing_signature != expected_signature:
        raise ValueError(
            "An existing checkpoint was made with different input or "
            f"settings: {json_path}. Use a new --output-dir for the changed "
            "configuration."
        )
    if metadata.get("status") != "completed":
        return None
    return metadata


def process_one_taxon(
    query_row: pd.Series,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    taxon_key = clean_int(query_row["gbif_occurrence_taxonKey"])
    if taxon_key is None:
        raise ValueError("Invalid taxonKey in query table")

    signature = config_signature(query_row, args)
    csv_path, json_path = checkpoint_paths(checkpoint_dir, taxon_key)
    existing = load_valid_checkpoint(csv_path, json_path, signature)
    if existing is not None:
        return existing

    raw, api_total = download_taxon_occurrences(
        query_row,
        max_fetch=args.max_fetch_per_species,
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
    )
    final, qc_stats = quality_filter_and_sample(
        raw,
        taxon_key=taxon_key,
        max_uncertainty_m=args.max_coordinate_uncertainty_m,
        coordinate_decimals=args.coordinate_decimals,
        max_final=args.max_final_per_species,
        grid_degrees=args.balance_grid_degrees,
    )

    metadata: dict[str, Any] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "config_signature": signature,
        "gbif_occurrence_taxonKey": taxon_key,
        "resolved_scientific_name": clean_text(
            query_row["resolved_scientific_name"]
        ),
        "input_species_names": clean_text(
            query_row["input_species_names"]
        ),
        "missing_records_n": clean_int(query_row["missing_records_n"]) or 0,
        "gbif_statuses": clean_text(query_row.get("gbif_statuses")),
        "taxonomic_caution": (
            "GBIF_DOUBTFUL"
            if "DOUBTFUL" in clean_text(
                query_row.get("gbif_statuses")
            ).upper()
            else ""
        ),
        "api_total_records": api_total,
        "api_records_fetched": len(raw),
        "api_truncated": api_total > len(raw),
        "max_fetch_per_species": args.max_fetch_per_species,
        "max_final_per_species": args.max_final_per_species,
        "max_coordinate_uncertainty_m": (
            args.max_coordinate_uncertainty_m
        ),
        "coordinate_decimals": args.coordinate_decimals,
        "balance_grid_degrees": args.balance_grid_degrees,
        **qc_stats,
    }

    # Write the record file first and metadata last. The metadata file acts as
    # the completion marker.
    atomic_write_csv(final, csv_path)
    atomic_write_json(metadata, json_path)
    return metadata


def collect_completed_outputs(
    query: pd.DataFrame,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    summaries: list[dict[str, Any]] = []
    record_frames: list[pd.DataFrame] = []
    incomplete_keys: list[int] = []

    for _, row in query.iterrows():
        taxon_key = clean_int(row["gbif_occurrence_taxonKey"])
        if taxon_key is None:
            continue
        signature = config_signature(row, args)
        csv_path, json_path = checkpoint_paths(checkpoint_dir, taxon_key)
        metadata = load_valid_checkpoint(
            csv_path,
            json_path,
            signature,
        )
        if metadata is None:
            incomplete_keys.append(taxon_key)
            continue

        summaries.append(metadata)
        if csv_path.stat().st_size > 0:
            try:
                records = pd.read_csv(csv_path, low_memory=False)
            except pd.errors.EmptyDataError:
                records = pd.DataFrame()
            if not records.empty:
                record_frames.append(records)

    summary = pd.DataFrame(summaries)
    records = (
        pd.concat(record_frames, ignore_index=True, sort=False)
        if record_frames
        else pd.DataFrame()
    )
    return records, summary, incomplete_keys


def build_overall_summary(
    query: pd.DataFrame,
    species_summary: pd.DataFrame,
    records: pd.DataFrame,
    failed_count: int,
) -> pd.DataFrame:
    completed = len(species_summary)
    with_api_records = (
        int((species_summary["api_total_records"] > 0).sum())
        if completed
        else 0
    )
    with_final_records = (
        int((species_summary["final_sample_records"] > 0).sum())
        if completed
        else 0
    )
    input_missing_completed = (
        int(species_summary["missing_records_n"].sum())
        if completed
        else 0
    )
    input_missing_with_range = (
        int(
            species_summary.loc[
                species_summary["final_sample_records"] > 0,
                "missing_records_n",
            ].sum()
        )
        if completed
        else 0
    )
    record_weighted_coverage = (
        input_missing_with_range / input_missing_completed
        if input_missing_completed
        else float("nan")
    )

    rows = [
        ("input_taxonKeys", len(query)),
        ("completed_taxonKeys", completed),
        ("failed_taxonKeys_this_run", failed_count),
        ("taxonKeys_with_api_records", with_api_records),
        ("taxonKeys_with_final_records", with_final_records),
        ("taxonKeys_without_final_records", completed - with_final_records),
        ("final_occurrence_records", len(records)),
        (
            "completed_input_missing_records_n",
            input_missing_completed,
        ),
        (
            "input_missing_records_with_GBIF_range_n",
            input_missing_with_range,
        ),
        (
            "record_weighted_GBIF_range_coverage",
            record_weighted_coverage,
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def validate_args(args: argparse.Namespace) -> None:
    if args.max_fetch_per_species < 1:
        raise ValueError("--max-fetch-per-species must be at least 1")
    if args.max_final_per_species < 1:
        raise ValueError("--max-final-per-species must be at least 1")
    if args.max_final_per_species > args.max_fetch_per_species:
        raise ValueError(
            "--max-final-per-species cannot exceed "
            "--max-fetch-per-species"
        )
    if args.max_coordinate_uncertainty_m < 0:
        raise ValueError(
            "--max-coordinate-uncertainty-m cannot be negative"
        )
    if not 0 <= args.coordinate_decimals <= 10:
        raise ValueError("--coordinate-decimals must be between 0 and 10")
    if args.balance_grid_degrees <= 0:
        raise ValueError("--balance-grid-degrees must be greater than 0")
    if args.max_taxa is not None and args.max_taxa < 1:
        raise ValueError("--max-taxa must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")
    if args.request_delay < 0:
        raise ValueError("--request-delay cannot be negative")


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "occurrence_by_taxon"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("=" * 100)
    print("LOADING FINAL SPECIES QUERY TABLE")
    print("=" * 100)
    query = pd.read_csv(input_path, low_memory=False)
    require_columns(query, INPUT_REQUIRED_COLUMNS, "Species query table")
    query["gbif_occurrence_taxonKey"] = query[
        "gbif_occurrence_taxonKey"
    ].map(clean_int)
    query["missing_records_n"] = (
        pd.to_numeric(query["missing_records_n"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )

    if query["gbif_occurrence_taxonKey"].isna().any():
        raise ValueError("The query table contains missing/invalid taxonKeys")
    if query["gbif_occurrence_taxonKey"].duplicated().any():
        duplicates = query.loc[
            query["gbif_occurrence_taxonKey"].duplicated(keep=False),
            "gbif_occurrence_taxonKey",
        ].tolist()
        raise ValueError(
            "The query table contains duplicate taxonKeys: "
            + ", ".join(str(value) for value in sorted(set(duplicates)))
        )

    query = query.sort_values(
        ["missing_records_n", "gbif_occurrence_taxonKey"],
        ascending=[False, True],
    ).reset_index(drop=True)
    print(f"Unique taxonKeys loaded: {len(query):,}")
    print(
        "Maximum API records per taxon: "
        f"{args.max_fetch_per_species:,}"
    )
    print(
        "Maximum final records per taxon: "
        f"{args.max_final_per_species:,}"
    )
    print(
        "Known uncertainty threshold: "
        f"{args.max_coordinate_uncertainty_m:,.0f} m"
    )

    unfinished_rows: list[pd.Series] = []
    already_completed = 0
    for _, row in query.iterrows():
        taxon_key = clean_int(row["gbif_occurrence_taxonKey"])
        signature = config_signature(row, args)
        csv_path, json_path = checkpoint_paths(
            checkpoint_dir,
            taxon_key or -1,
        )
        if load_valid_checkpoint(csv_path, json_path, signature):
            already_completed += 1
        else:
            unfinished_rows.append(row)

    if args.max_taxa is not None:
        unfinished_rows = unfinished_rows[: args.max_taxa]

    print(f"Already completed checkpoints: {already_completed:,}")
    print(f"Taxa selected for this run:    {len(unfinished_rows):,}")

    print("\n" + "=" * 100)
    print("QUERYING AND FILTERING GBIF OCCURRENCES")
    print("=" * 100)
    failures: list[dict[str, Any]] = []

    for position, row in enumerate(unfinished_rows, start=1):
        taxon_key = clean_int(row["gbif_occurrence_taxonKey"])
        name = clean_text(row["resolved_scientific_name"])
        print(
            f"[{position}/{len(unfinished_rows)}] "
            f"taxonKey {taxon_key} | {name}"
        )
        try:
            metadata = process_one_taxon(
                row,
                args=args,
                checkpoint_dir=checkpoint_dir,
            )
            print(
                "    API total "
                f"{metadata['api_total_records']:,} | "
                f"fetched {metadata['records_fetched']:,} | "
                f"after QC {metadata['records_after_local_qc']:,} | "
                f"unique coords {metadata['unique_coordinate_records']:,} | "
                f"final {metadata['final_sample_records']:,}"
            )
        except Exception as exc:
            failures.append(
                {
                    "gbif_occurrence_taxonKey": taxon_key,
                    "resolved_scientific_name": name,
                    "input_species_names": clean_text(
                        row["input_species_names"]
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at_utc": utc_now(),
                }
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 100)
    print("COMBINING COMPLETED CHECKPOINTS")
    print("=" * 100)
    records, species_summary, incomplete_keys = collect_completed_outputs(
        query,
        args=args,
        checkpoint_dir=checkpoint_dir,
    )

    records_path = output_dir / "gbif_occurrence_records_for_realm.csv"
    species_summary_path = (
        output_dir / "gbif_occurrence_species_summary.csv"
    )
    pilot_summary_path = (
        output_dir / "gbif_occurrence_pilot_summary.csv"
    )
    failed_path = output_dir / "gbif_occurrence_failed_taxa.csv"

    if not species_summary.empty:
        species_summary = species_summary.sort_values(
            ["final_sample_records", "missing_records_n"],
            ascending=[False, False],
        ).reset_index(drop=True)

    failed = pd.DataFrame(failures, columns=FAILED_COLUMNS)
    overall = build_overall_summary(
        query,
        species_summary,
        records,
        failed_count=len(failed),
    )

    atomic_write_csv(records, records_path)
    atomic_write_csv(species_summary, species_summary_path)
    atomic_write_csv(overall, pilot_summary_path)
    atomic_write_csv(failed, failed_path)

    completed_n = len(species_summary)
    final_positive_n = (
        int((species_summary["final_sample_records"] > 0).sum())
        if completed_n
        else 0
    )

    print(f"Completed taxonKeys:            {completed_n:,}/{len(query):,}")
    print(f"TaxonKeys with final records:   {final_positive_n:,}")
    print(f"Final occurrence records:       {len(records):,}")
    print(f"Failures in this run:           {len(failed):,}")
    print(f"Still incomplete taxonKeys:     {len(incomplete_keys):,}")

    print("\nSaved files:")
    for path in [
        records_path,
        species_summary_path,
        pilot_summary_path,
        failed_path,
    ]:
        print(path)
    print(f"Per-taxon checkpoints: {checkpoint_dir}")

    if incomplete_keys:
        print(
            "\nThe pilot is partially complete. Run the same command again "
            "to resume unfinished taxa."
        )
    else:
        print(
            "\nGBIF occurrence pilot completed successfully for all taxonKeys."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
