#!/usr/bin/env python3
"""
Validate GBIF species-range realm predictions against independent internal records.

The validation truth is restricted to records with an original coordinate and an
existing realm label. Coordinates added by geocoding or centroid assignment are
never used as truth.

The script:
  1. reads the GBIF species-level realm/confidence table;
  2. streams the large master CSV and retains original-coordinate truth records
     for pilot species only;
  3. reports record-level and species-balanced accuracy;
  4. audits the existing High/Medium/Low confidence bands;
  5. uses a species-level calibration/holdout split to search possible dominant-
     realm-proportion and minimum-GBIF-record thresholds.

Only pandas and numpy are required.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_MASTER = "centroid_results/new_all_seq_with_final_centroids.csv"
DEFAULT_GBIF = (
    "gbif_species_range_pilot/gbif_realm_assignment/"
    "gbif_species_realm_confidence.csv"
)
DEFAULT_OUTPUT = (
    "gbif_species_range_pilot/gbif_species_range_accuracy_validation"
)

MISSING_STRINGS = {"", "na", "nan", "none", "null", "<na>"}
NON_TERRESTRIAL = {
    "marine",
    "ocean",
    "oceanic",
    "sea",
    "water",
    "freshwater",
    "unassigned",
    "unknown",
    "outside",
}

REALM_CANONICAL = {
    "afrotropic": "Afrotropical",
    "afrotropical": "Afrotropical",
    "afrotropics": "Afrotropical",
    "australasia": "Australasian",
    "australasian": "Australasian",
    "indo malayan": "Indomalayan",
    "indo malaya": "Indomalayan",
    "indomalaya": "Indomalayan",
    "indomalayan": "Indomalayan",
    "nearctic": "Nearctic",
    "neotropic": "Neotropical",
    "neotropical": "Neotropical",
    "oceania": "Oceanian",
    "oceanian": "Oceanian",
    "palearctic": "Palearctic",
    "palaearctic": "Palearctic",
    "panamanian": "Panamanian",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GBIF dominant-realm predictions and audit the current "
            "High/Medium/Low pilot_realm_confidence thresholds."
        )
    )
    parser.add_argument("master_csv", nargs="?", default=DEFAULT_MASTER)
    parser.add_argument("--gbif-confidence", default=DEFAULT_GBIF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--high-target", type=float, default=0.90)
    parser.add_argument("--medium-target", type=float, default=0.75)
    parser.add_argument("--min-band-species", type=int, default=10)
    parser.add_argument(
        "--dominance-grid",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated dominant-realm proportions to test.",
    )
    parser.add_argument(
        "--minimum-n-grid",
        default="5,10,20,30,50",
        help="Comma-separated minimum assigned GBIF occurrence counts to test.",
    )
    parser.add_argument(
        "--coverage-grid",
        default="0.50,0.70,0.80,0.90",
        help=(
            "Comma-separated minimum numeric overlay-coverage proportions to "
            "test. Ignored if no numeric overlay_coverage column exists."
        ),
    )
    return parser.parse_args()


def clean_text(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return None if text.casefold() in MISSING_STRINGS else text


def species_key(value: object) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).casefold()


def split_input_species_names(value: object) -> list[Optional[str]]:
    """Return every original species name represented by one GBIF query row."""
    text = clean_text(value)
    if text is None:
        return [None]

    # Support JSON/Python-style lists if the CSV stored multiple input names
    # as ["Species a", "Species b"] or ['Species a', 'Species b'].
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                names = [clean_text(item) for item in parsed]
                names = [name for name in names if name is not None]
                if names:
                    return names
        except (SyntaxError, ValueError):
            pass

    # Commas are deliberately not separators because scientific names may
    # contain authorship text with commas.
    names = [
        clean_text(item)
        for item in re.split(r"\s*(?:\||;|\n)\s*", text)
    ]
    names = [name for name in names if name is not None]
    return names or [None]


def normalise_realm(value: object) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    key = re.sub(r"[_\-]+", " ", text).casefold().strip()
    key = re.sub(r"\s+", " ", key)
    if key in NON_TERRESTRIAL:
        return None
    return REALM_CANONICAL.get(key, text.strip().title())


def normalise_confidence(value: object) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    key = text.casefold()
    if key.startswith("high"):
        return "High"
    if key.startswith("med"):
        return "Medium"
    if key.startswith("low"):
        return "Low"
    return text


def find_column(
    columns: Iterable[str],
    aliases: Iterable[str],
    *,
    required: bool,
    label: str,
) -> Optional[str]:
    columns = list(columns)
    exact = {str(c).casefold(): c for c in columns}
    for alias in aliases:
        if alias.casefold() in exact:
            return exact[alias.casefold()]
    if required:
        raise ValueError(
            f"Could not identify {label}. Tried: {list(aliases)}\n"
            f"Available columns: {columns}"
        )
    return None


def numeric_fraction(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if not finite.empty and finite.quantile(0.75) > 1.0:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)


def resolve_gbif_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    columns = df.columns
    return {
        "species": find_column(
            columns,
            [
                "species",
                "input_species",
                "input_species_names",
                "pilot_species",
                "query_name",
                "query_scientific_name",
            ],
            required=True,
            label="pilot species column",
        ),
        "predicted_realm": find_column(
            columns,
            [
                "dominant_realm",
                "primary_realm",
                "pilot_realm",
                "inferred_realm",
                "predicted_realm",
                "gbif_dominant_realm",
                "top_realm",
            ],
            required=True,
            label="GBIF dominant/predicted realm column",
        ),
        "confidence": find_column(
            columns,
            [
                "pilot_realm_confidence",
                "realm_confidence",
                "confidence_class",
                "confidence_level",
            ],
            required=True,
            label="pilot_realm_confidence column",
        ),
        "dominance": find_column(
            columns,
            [
                "dominant_realm_proportion",
                "dominant_realm_fraction",
                "dominant_fraction",
                "dominant_share",
                "top_realm_proportion",
                "dominant_realm_pct",
                "dominant_realm_percentage",
            ],
            required=True,
            label="dominant realm proportion column",
        ),
        "assigned_n": find_column(
            columns,
            [
                "assigned_realm_records_n",
                "realm_assigned_records_n",
                "assigned_records_n",
                "valid_realm_records_n",
                "terrestrial_records_n",
                "realm_records_n",
                "total_assigned_n",
                "gbif_realm_records_n",
                "assigned_occurrences_n",
            ],
            required=False,
            label="number of assigned GBIF occurrences",
        ),
        "coverage": find_column(
            columns,
            [
                "overlay_coverage",
                "overlay_coverage_proportion",
                "realm_assignment_rate",
                "realm_overlay_coverage",
                "realm_assignment_coverage",
            ],
            required=False,
            label="overlay coverage proportion",
        ),
        "coverage_quality": find_column(
            columns,
            ["overlay_coverage_quality", "coverage_quality"],
            required=False,
            label="overlay coverage quality",
        ),
    }


def prepare_gbif_table(path: Path) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    raw = pd.read_csv(path, low_memory=False)
    mapping = resolve_gbif_columns(raw)
    raw = raw.assign(
        _validation_species=raw[mapping["species"]].map(
            split_input_species_names
        )
    ).explode("_validation_species", ignore_index=True)

    out = pd.DataFrame(
        {
            "species": raw["_validation_species"].map(clean_text),
            "species_key": raw["_validation_species"].map(species_key),
            "predicted_realm": raw[mapping["predicted_realm"]].map(
                normalise_realm
            ),
            "pilot_realm_confidence": raw[mapping["confidence"]].map(
                normalise_confidence
            ),
            "dominant_realm_proportion": numeric_fraction(
                raw[mapping["dominance"]]
            ),
        }
    )

    assigned_col = mapping["assigned_n"]
    out["gbif_assigned_realm_records_n"] = (
        pd.to_numeric(raw[assigned_col], errors="coerce")
        if assigned_col
        else np.nan
    )
    coverage_col = mapping["coverage"]
    out["overlay_coverage"] = (
        numeric_fraction(raw[coverage_col]) if coverage_col else np.nan
    )
    quality_col = mapping["coverage_quality"]
    out["overlay_coverage_quality"] = (
        raw[quality_col].map(clean_text) if quality_col else None
    )

    out = out.dropna(subset=["species_key"]).copy()
    duplicates = out[out.duplicated("species_key", keep=False)]
    if not duplicates.empty:
        conflicting = (
            duplicates.groupby("species_key")["predicted_realm"]
            .nunique(dropna=True)
            .gt(1)
        )
        bad_keys = set(conflicting[conflicting].index)
        if bad_keys:
            examples = sorted(bad_keys)[:10]
            raise ValueError(
                "GBIF confidence table has conflicting dominant realms for "
                f"{len(bad_keys)} species. Examples: {examples}"
            )
        out = out.sort_values(
            ["species_key", "gbif_assigned_realm_records_n"],
            ascending=[True, False],
            na_position="last",
        ).drop_duplicates("species_key", keep="first")

    return out.reset_index(drop=True), mapping


def resolve_master_columns(header: pd.DataFrame) -> dict[str, Optional[str]]:
    columns = header.columns
    return {
        "species": find_column(
            columns,
            ["species", "scientific_name", "scientificName"],
            required=True,
            label="master species column",
        ),
        "realm": find_column(
            columns,
            ["realm", "original_realm", "biogeographic_realm"],
            required=True,
            label="master realm column",
        ),
        "original_lat": find_column(
            columns,
            ["original_latitude", "original_lat", "latitude_original"],
            required=False,
            label="original latitude",
        ),
        "original_lon": find_column(
            columns,
            ["original_longitude", "original_lon", "longitude_original"],
            required=False,
            label="original longitude",
        ),
        "source": find_column(
            columns,
            ["final_geocode_source", "coordinate_source", "geocode_source"],
            required=False,
            label="final coordinate source",
        ),
        "valid": find_column(
            columns,
            ["final_coordinate_valid", "coordinate_valid"],
            required=False,
            label="coordinate validity",
        ),
        "record_id": find_column(
            columns,
            ["record_id", "db_id", "BTseq_id", "specimen_id", "sampleid"],
            required=False,
            label="record identifier",
        ),
    }


def truth_mask(chunk: pd.DataFrame, mapping: dict[str, Optional[str]]) -> pd.Series:
    realm_ok = chunk[mapping["realm"]].map(normalise_realm).notna()
    source_col = mapping["source"]
    lat_col = mapping["original_lat"]
    lon_col = mapping["original_lon"]

    if lat_col and lon_col:
        lat = pd.to_numeric(chunk[lat_col], errors="coerce")
        lon = pd.to_numeric(chunk[lon_col], errors="coerce")
        original_coordinate = (
            lat.between(-90, 90, inclusive="both")
            & lon.between(-180, 180, inclusive="both")
        )
    elif source_col:
        original_coordinate = (
            chunk[source_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("original")
        )
    else:
        raise ValueError(
            "Master CSV needs either original_latitude/original_longitude or "
            "final_geocode_source so original-coordinate truth can be isolated."
        )

    valid_col = mapping["valid"]
    if valid_col:
        valid_text = (
            chunk[valid_col].fillna(False).astype(str).str.casefold().str.strip()
        )
        coordinate_valid = valid_text.isin({"true", "1", "yes", "y"})
        original_coordinate &= coordinate_valid

    return realm_ok & original_coordinate


def read_validation_truth(
    master_path: Path,
    pilot_keys: set[str],
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, Optional[str]], dict[str, int]]:
    header = pd.read_csv(master_path, nrows=0)
    mapping = resolve_master_columns(header)
    usecols = list(dict.fromkeys(c for c in mapping.values() if c is not None))

    retained: list[pd.DataFrame] = []
    total_rows = 0
    original_truth_rows = 0
    pilot_truth_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            master_path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)
        keys = chunk[mapping["species"]].map(species_key)
        mask = truth_mask(chunk, mapping)
        original_truth_rows += int(mask.sum())
        keep = mask & keys.isin(pilot_keys)
        pilot_truth_rows += int(keep.sum())

        if keep.any():
            selected = pd.DataFrame(
                {
                    "species_key": keys[keep],
                    "truth_species": chunk.loc[keep, mapping["species"]].map(
                        clean_text
                    ),
                    "observed_realm": chunk.loc[keep, mapping["realm"]].map(
                        normalise_realm
                    ),
                }
            )
            record_col = mapping["record_id"]
            if record_col:
                selected["truth_record_id"] = chunk.loc[keep, record_col].astype(
                    "string"
                )
            else:
                selected["truth_record_id"] = [
                    f"row_{total_rows - len(chunk) + i + 1}"
                    for i in np.flatnonzero(keep.to_numpy())
                ]
            retained.append(selected)

        print(
            f"Chunk {chunk_number}: {len(chunk):,} rows | "
            f"original-coordinate truth {int(mask.sum()):,} | "
            f"pilot truth {int(keep.sum()):,}"
        )

    truth = (
        pd.concat(retained, ignore_index=True)
        if retained
        else pd.DataFrame(
            columns=[
                "species_key",
                "truth_species",
                "observed_realm",
                "truth_record_id",
            ]
        )
    )
    inventory = {
        "master_rows": total_rows,
        "all_original_coordinate_truth_rows": original_truth_rows,
        "pilot_original_coordinate_truth_rows": pilot_truth_rows,
    }
    return truth, mapping, inventory


def majority_realm(series: pd.Series) -> Optional[str]:
    counts = series.dropna().value_counts()
    if counts.empty:
        return None
    top_n = counts.iloc[0]
    tied = sorted(counts[counts.eq(top_n)].index.astype(str))
    return tied[0]


def wilson_interval(successes: float, total: float) -> tuple[float, float]:
    if total <= 0 or not np.isfinite(successes):
        return np.nan, np.nan
    p = successes / total
    z = 1.959963984540054
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_macro_ci(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or repetitions <= 0:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    batch = max(1, min(250, repetitions))
    position = 0
    while position < repetitions:
        n_now = min(batch, repetitions - position)
        indices = rng.integers(0, values.size, size=(n_now, values.size))
        means[position : position + n_now] = values[indices].mean(axis=1)
        position += n_now
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def build_species_metrics(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby("species_key", sort=True, observed=True)
    metrics = grouped.agg(
        species=("species", "first"),
        predicted_realm=("predicted_realm", "first"),
        pilot_realm_confidence=("pilot_realm_confidence", "first"),
        dominant_realm_proportion=("dominant_realm_proportion", "first"),
        gbif_assigned_realm_records_n=(
            "gbif_assigned_realm_records_n",
            "first",
        ),
        overlay_coverage=("overlay_coverage", "first"),
        overlay_coverage_quality=("overlay_coverage_quality", "first"),
        validation_records_n=("is_correct", "size"),
        correct_validation_records_n=("is_correct", "sum"),
        observed_realms_n=("observed_realm", "nunique"),
        observed_dominant_realm=("observed_realm", majority_realm),
    ).reset_index()
    metrics["record_match_rate"] = (
        metrics["correct_validation_records_n"]
        / metrics["validation_records_n"]
    )
    metrics["observed_dominant_realm_matches"] = (
        metrics["observed_dominant_realm"] == metrics["predicted_realm"]
    )
    return metrics


def accuracy_row(
    label: str,
    subset: pd.DataFrame,
    *,
    total_species: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    n_species = int(len(subset))
    n_records = int(subset["validation_records_n"].sum()) if n_species else 0
    correct_records = (
        int(subset["correct_validation_records_n"].sum()) if n_species else 0
    )
    micro = correct_records / n_records if n_records else np.nan
    micro_low, micro_high = wilson_interval(correct_records, n_records)
    macro = subset["record_match_rate"].mean() if n_species else np.nan
    macro_low, macro_high = bootstrap_macro_ci(
        subset["record_match_rate"].to_numpy(dtype=float),
        repetitions=bootstrap,
        seed=seed,
    )
    dominant_accuracy = (
        subset["observed_dominant_realm_matches"].mean()
        if n_species
        else np.nan
    )
    dominant_low, dominant_high = wilson_interval(
        int(subset["observed_dominant_realm_matches"].sum()) if n_species else 0,
        n_species,
    )
    return {
        "group": label,
        "validation_species_n": n_species,
        "validation_records_n": n_records,
        "species_coverage": n_species / total_species if total_species else np.nan,
        "record_level_accuracy_micro": micro,
        "record_level_accuracy_micro_ci95_low": micro_low,
        "record_level_accuracy_micro_ci95_high": micro_high,
        "species_balanced_accuracy_macro": macro,
        "species_balanced_accuracy_macro_ci95_low": macro_low,
        "species_balanced_accuracy_macro_ci95_high": macro_high,
        "species_dominant_realm_accuracy": dominant_accuracy,
        "species_dominant_realm_accuracy_ci95_low": dominant_low,
        "species_dominant_realm_accuracy_ci95_high": dominant_high,
    }


def stratified_species_split(
    species_df: pd.DataFrame,
    calibration_fraction: float,
    seed: int,
) -> pd.Series:
    if not 0 < calibration_fraction < 1:
        raise ValueError("--calibration-fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    split = pd.Series("holdout", index=species_df.index, dtype="string")
    strata = species_df["pilot_realm_confidence"].fillna("Missing")
    for _, indices in strata.groupby(strata).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        rng.shuffle(indices)
        if indices.size == 1:
            calibration_n = 1
        else:
            calibration_n = int(round(indices.size * calibration_fraction))
            calibration_n = min(max(calibration_n, 1), indices.size - 1)
        split.loc[indices[:calibration_n]] = "calibration"
    return split


def evaluate_threshold_grid(
    species_df: pd.DataFrame,
    proportions: list[float],
    minimum_ns: list[int],
    minimum_coverages: list[float],
    *,
    total_species: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for proportion in proportions:
        for minimum_n in minimum_ns:
            for minimum_coverage in minimum_coverages:
                n_values = species_df["gbif_assigned_realm_records_n"]
                n_ok = (
                    pd.Series(True, index=species_df.index)
                    if minimum_n == 0 and n_values.isna().all()
                    else n_values.ge(minimum_n)
                )
                coverage_values = species_df["overlay_coverage"]
                coverage_ok = (
                    pd.Series(True, index=species_df.index)
                    if minimum_coverage == 0 and coverage_values.isna().all()
                    else coverage_values.ge(minimum_coverage)
                )
                subset = species_df[
                    species_df["dominant_realm_proportion"].ge(proportion)
                    & n_ok
                    & coverage_ok
                ]
                n_species = len(subset)
                successes = int(
                    subset["observed_dominant_realm_matches"].sum()
                )
                accuracy = successes / n_species if n_species else np.nan
                ci_low, ci_high = wilson_interval(successes, n_species)
                rows.append(
                    {
                        "dominant_realm_proportion_threshold": proportion,
                        "minimum_assigned_gbif_records": minimum_n,
                        "minimum_overlay_coverage": minimum_coverage,
                        "validation_species_n": n_species,
                        "species_coverage": (
                            n_species / total_species
                            if total_species
                            else np.nan
                        ),
                        "species_dominant_realm_accuracy": accuracy,
                        "accuracy_ci95_low": ci_low,
                        "accuracy_ci95_high": ci_high,
                        "species_balanced_record_accuracy": (
                            subset["record_match_rate"].mean()
                            if n_species
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def select_candidate(
    grid: pd.DataFrame,
    target: float,
    min_species: int,
) -> tuple[Optional[pd.Series], str]:
    eligible = grid[
        grid["validation_species_n"].ge(min_species)
        & grid["accuracy_ci95_low"].ge(target)
    ].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(
            [
                "species_coverage",
                "dominant_realm_proportion_threshold",
                "minimum_assigned_gbif_records",
                "minimum_overlay_coverage",
            ],
            ascending=[False, True, True, True],
        )
        return eligible.iloc[0], "meets_target_with_ci"

    fallback = grid[grid["validation_species_n"].ge(min_species)].copy()
    if fallback.empty:
        return None, "insufficient_validation_species"
    fallback = fallback.sort_values(
        [
            "accuracy_ci95_low",
            "species_dominant_realm_accuracy",
            "species_coverage",
        ],
        ascending=[False, False, False],
    )
    return fallback.iloc[0], "best_available_but_target_not_confirmed"


def audit_band(
    row: Optional[pd.Series],
    target: float,
    min_species: int,
) -> str:
    if row is None or int(row["validation_species_n"]) < min_species:
        return "INSUFFICIENT_EVIDENCE"
    if row["species_dominant_realm_accuracy_ci95_low"] >= target:
        return "RETAIN"
    if row["species_dominant_realm_accuracy"] >= target:
        return "PROVISIONAL_MORE_VALIDATION_NEEDED"
    return "MODIFY_OR_RESTRICT_AUTOMATIC_USE"


def format_number(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    master_path = Path(args.master_csv)
    gbif_path = Path(args.gbif_confidence)
    output_dir = Path(args.output_dir)

    if not master_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_path}")
    if not gbif_path.exists():
        raise FileNotFoundError(f"GBIF confidence CSV not found: {gbif_path}")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive.")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative.")

    proportions = sorted(
        {
            float(x.strip())
            for x in args.dominance_grid.split(",")
            if x.strip()
        }
    )
    minimum_ns = sorted(
        {int(x.strip()) for x in args.minimum_n_grid.split(",") if x.strip()}
    )
    minimum_coverages = sorted(
        {
            float(x.strip())
            for x in args.coverage_grid.split(",")
            if x.strip()
        }
    )
    if not proportions or any(x < 0 or x > 1 for x in proportions):
        raise ValueError("--dominance-grid values must be within 0 to 1.")
    if not minimum_ns or any(x < 0 for x in minimum_ns):
        raise ValueError("--minimum-n-grid values must be non-negative.")
    if (
        not minimum_coverages
        or any(x < 0 or x > 1 for x in minimum_coverages)
    ):
        raise ValueError("--coverage-grid values must be within 0 to 1.")

    print("=" * 100)
    print("GBIF SPECIES-RANGE REALM ACCURACY VALIDATION")
    print("=" * 100)
    print(f"Master CSV:       {master_path}")
    print(f"GBIF confidence:  {gbif_path}")
    print(f"Output directory: {output_dir}")

    gbif, gbif_mapping = prepare_gbif_table(gbif_path)
    if gbif_mapping["assigned_n"] is None:
        minimum_ns = [0]
        print(
            "\nNote: no assigned-GBIF-occurrence-count column was detected; "
            "no minimum occurrence-count threshold will be invented."
        )
    if gbif_mapping["coverage"] is None:
        minimum_coverages = [0.0]
        print(
            "\nNote: no numeric overlay_coverage column was detected; "
            "overlay_coverage_quality will remain in diagnostic outputs, but "
            "no numeric coverage threshold will be invented."
        )
    pilot_keys = set(gbif["species_key"])
    print(f"\nPilot species in confidence table: {len(gbif):,}")
    print("Detected GBIF columns:")
    for role, column in gbif_mapping.items():
        print(f"  {role:20s}: {column or 'not available'}")

    truth, master_mapping, inventory = read_validation_truth(
        master_path, pilot_keys, args.chunksize
    )
    print("\nDetected master columns:")
    for role, column in master_mapping.items():
        print(f"  {role:20s}: {column or 'not available'}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_evidence = (
        truth.groupby("species_key", observed=True)
        .agg(
            original_validation_records_n=("observed_realm", "size"),
            original_validation_realms_n=("observed_realm", "nunique"),
            original_validation_dominant_realm=(
                "observed_realm",
                majority_realm,
            ),
        )
        .reset_index()
    )
    coverage_audit = gbif.merge(pilot_evidence, on="species_key", how="left")
    coverage_audit["has_validation_truth"] = (
        coverage_audit["original_validation_records_n"].fillna(0).gt(0)
    )
    coverage_audit.to_csv(
        output_dir / "gbif_species_range_validation_coverage.csv", index=False
    )

    valid_predictions = gbif.dropna(subset=["predicted_realm"]).copy()
    records = truth.merge(valid_predictions, on="species_key", how="inner")
    if records.empty:
        raise RuntimeError(
            "No validation records remained after matching pilot species with "
            "original-coordinate realm truth. Inspect "
            "gbif_species_range_validation_coverage.csv."
        )

    records["is_correct"] = (
        records["observed_realm"] == records["predicted_realm"]
    )
    records.to_csv(
        output_dir / "gbif_species_range_validation_records.csv", index=False
    )

    species_metrics = build_species_metrics(records)
    species_metrics["validation_split"] = stratified_species_split(
        species_metrics, args.calibration_fraction, args.seed
    )
    species_metrics.to_csv(
        output_dir / "gbif_species_range_validation_by_species.csv", index=False
    )

    total_validated_species = len(species_metrics)
    band_rows = [
        accuracy_row(
            "All",
            species_metrics,
            total_species=total_validated_species,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
    ]
    for index, band in enumerate(["High", "Medium", "Low"], start=1):
        band_rows.append(
            accuracy_row(
                band,
                species_metrics[
                    species_metrics["pilot_realm_confidence"].eq(band)
                ],
                total_species=total_validated_species,
                bootstrap=args.bootstrap,
                seed=args.seed + index,
            )
        )
    accuracy_by_band = pd.DataFrame(band_rows)
    accuracy_by_band.to_csv(
        output_dir / "gbif_species_range_accuracy_by_confidence.csv",
        index=False,
    )

    confusion = pd.crosstab(
        records["observed_realm"],
        records["predicted_realm"],
        margins=True,
        margins_name="Total",
    )
    confusion.to_csv(
        output_dir / "gbif_species_range_confusion_matrix_records.csv"
    )

    calibration = species_metrics[
        species_metrics["validation_split"].eq("calibration")
    ].copy()
    holdout = species_metrics[
        species_metrics["validation_split"].eq("holdout")
    ].copy()
    grid = evaluate_threshold_grid(
        calibration,
        proportions,
        minimum_ns,
        minimum_coverages,
        total_species=len(calibration),
    )
    grid.to_csv(
        output_dir / "gbif_species_range_threshold_grid_calibration.csv",
        index=False,
    )

    recommendation_rows: list[dict[str, object]] = []
    candidates: dict[str, tuple[Optional[pd.Series], str]] = {}
    for band, target in [
        ("High", args.high_target),
        ("Medium", args.medium_target),
    ]:
        candidate, status = select_candidate(
            grid, target, args.min_band_species
        )
        candidates[band] = (candidate, status)
        if candidate is None:
            recommendation_rows.append(
                {
                    "band": band,
                    "target_accuracy": target,
                    "selection_status": status,
                    "recommended_dominant_realm_proportion_threshold": np.nan,
                    "recommended_minimum_assigned_gbif_records": np.nan,
                    "recommended_minimum_overlay_coverage": np.nan,
                    "calibration_species_n": 0,
                    "calibration_species_coverage": np.nan,
                    "calibration_accuracy": np.nan,
                    "calibration_accuracy_ci95_low": np.nan,
                    "calibration_accuracy_ci95_high": np.nan,
                    "holdout_species_n": 0,
                    "holdout_species_coverage": np.nan,
                    "holdout_accuracy": np.nan,
                    "holdout_accuracy_ci95_low": np.nan,
                    "holdout_accuracy_ci95_high": np.nan,
                }
            )
            continue

        threshold = float(
            candidate["dominant_realm_proportion_threshold"]
        )
        minimum_n = int(candidate["minimum_assigned_gbif_records"])
        minimum_coverage = float(candidate["minimum_overlay_coverage"])
        holdout_n = holdout["gbif_assigned_realm_records_n"]
        holdout_coverage = holdout["overlay_coverage"]
        holdout_n_ok = (
            pd.Series(True, index=holdout.index)
            if minimum_n == 0 and holdout_n.isna().all()
            else holdout_n.ge(minimum_n)
        )
        holdout_coverage_ok = (
            pd.Series(True, index=holdout.index)
            if minimum_coverage == 0 and holdout_coverage.isna().all()
            else holdout_coverage.ge(minimum_coverage)
        )
        holdout_selected = holdout[
            holdout["dominant_realm_proportion"].ge(threshold)
            & holdout_n_ok
            & holdout_coverage_ok
        ]
        successes = int(
            holdout_selected["observed_dominant_realm_matches"].sum()
        )
        holdout_accuracy = (
            successes / len(holdout_selected)
            if len(holdout_selected)
            else np.nan
        )
        holdout_low, holdout_high = wilson_interval(
            successes, len(holdout_selected)
        )
        recommendation_rows.append(
            {
                "band": band,
                "target_accuracy": target,
                "selection_status": status,
                "recommended_dominant_realm_proportion_threshold": threshold,
                "recommended_minimum_assigned_gbif_records": minimum_n,
                "recommended_minimum_overlay_coverage": minimum_coverage,
                "calibration_species_n": int(
                    candidate["validation_species_n"]
                ),
                "calibration_species_coverage": candidate["species_coverage"],
                "calibration_accuracy": candidate[
                    "species_dominant_realm_accuracy"
                ],
                "calibration_accuracy_ci95_low": candidate[
                    "accuracy_ci95_low"
                ],
                "calibration_accuracy_ci95_high": candidate[
                    "accuracy_ci95_high"
                ],
                "holdout_species_n": len(holdout_selected),
                "holdout_species_coverage": (
                    len(holdout_selected) / len(holdout)
                    if len(holdout)
                    else np.nan
                ),
                "holdout_accuracy": holdout_accuracy,
                "holdout_accuracy_ci95_low": holdout_low,
                "holdout_accuracy_ci95_high": holdout_high,
            }
        )
    recommendations = pd.DataFrame(recommendation_rows)
    recommendations.to_csv(
        output_dir / "gbif_species_range_threshold_recommendation.csv",
        index=False,
    )

    high_row_df = accuracy_by_band[accuracy_by_band["group"].eq("High")]
    medium_row_df = accuracy_by_band[accuracy_by_band["group"].eq("Medium")]
    high_row = high_row_df.iloc[0] if not high_row_df.empty else None
    medium_row = medium_row_df.iloc[0] if not medium_row_df.empty else None
    high_decision = audit_band(
        high_row, args.high_target, args.min_band_species
    )
    medium_decision = audit_band(
        medium_row, args.medium_target, args.min_band_species
    )

    summary_rows = [
        {"metric": key, "value": value} for key, value in inventory.items()
    ]
    summary_rows.extend(
        [
            {"metric": "pilot_species_total", "value": len(gbif)},
            {
                "metric": "pilot_species_with_validation_truth",
                "value": int(coverage_audit["has_validation_truth"].sum()),
            },
            {
                "metric": "pilot_species_without_validation_truth",
                "value": int((~coverage_audit["has_validation_truth"]).sum()),
            },
            {
                "metric": "validated_species_with_prediction",
                "value": total_validated_species,
            },
            {
                "metric": "validated_records_with_prediction",
                "value": len(records),
            },
            {
                "metric": "calibration_species_n",
                "value": len(calibration),
            },
            {"metric": "holdout_species_n", "value": len(holdout)},
            {
                "metric": "current_high_threshold_decision",
                "value": high_decision,
            },
            {
                "metric": "current_medium_threshold_decision",
                "value": medium_decision,
            },
            {
                "metric": "low_band_recommended_use",
                "value": "DO_NOT_AUTOMATICALLY_ASSIGN; retain as uncertain",
            },
        ]
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "gbif_species_range_validation_summary.csv", index=False
    )

    all_row = accuracy_by_band.iloc[0]
    report_lines = [
        "GBIF SPECIES-RANGE REALM ACCURACY VALIDATION",
        "=" * 60,
        "",
        "Validation design",
        "-----------------",
        (
            "Truth: existing realm labels from records with original "
            "coordinates only."
        ),
        (
            "Prediction: GBIF species-level dominant realm; geocoded and "
            "centroid coordinates are excluded from truth."
        ),
        (
            "Threshold selection: species-level calibration subset; candidate "
            "thresholds are checked again on a held-out species subset."
        ),
        "",
        "Coverage",
        "--------",
        f"Pilot species total: {len(gbif):,}",
        (
            "Pilot species with usable original-coordinate truth: "
            f"{int(coverage_audit['has_validation_truth'].sum()):,}"
        ),
        f"Validated records: {len(records):,}",
        "",
        "Overall accuracy",
        "----------------",
        (
            "Record-level accuracy: "
            f"{format_number(all_row['record_level_accuracy_micro'])} "
            f"(95% CI "
            f"{format_number(all_row['record_level_accuracy_micro_ci95_low'])}"
            "–"
            f"{format_number(all_row['record_level_accuracy_micro_ci95_high'])})"
        ),
        (
            "Species-balanced accuracy: "
            f"{format_number(all_row['species_balanced_accuracy_macro'])} "
            f"(bootstrap 95% CI "
            f"{format_number(all_row['species_balanced_accuracy_macro_ci95_low'])}"
            "–"
            f"{format_number(all_row['species_balanced_accuracy_macro_ci95_high'])})"
        ),
        "",
        "Current confidence-band decisions",
        "---------------------------------",
        f"High:   {high_decision}",
        f"Medium: {medium_decision}",
        "Low:    do not use for automatic assignment",
        "",
        "Interpretation rule",
        "-------------------",
        (
            "RETAIN means the lower 95% confidence bound reaches the requested "
            "accuracy target with enough validated species."
        ),
        (
            "PROVISIONAL means the point estimate reaches the target but the "
            "sample is not yet precise enough."
        ),
        (
            "MODIFY means the current band failed its target; inspect the "
            "recommended threshold table before changing production settings."
        ),
        (
            "INSUFFICIENT_EVIDENCE means this pilot cannot support a threshold "
            "change yet."
        ),
        "",
        "Threshold recommendation",
        "------------------------",
    ]
    for _, row in recommendations.iterrows():
        report_lines.append(
            f"{row['band']}: status={row['selection_status']}; "
            "dominance>="
            f"{format_number(row['recommended_dominant_realm_proportion_threshold'], 2)}; "
            "GBIF assigned n>="
            f"{format_number(row['recommended_minimum_assigned_gbif_records'], 0)}; "
            "overlay coverage>="
            f"{format_number(row['recommended_minimum_overlay_coverage'], 2)}; "
            "holdout accuracy="
            f"{format_number(row['holdout_accuracy'])}; "
            f"holdout species n={int(row['holdout_species_n'])}"
        )
    report_lines.extend(
        [
            "",
            "Important limitations",
            "---------------------",
            (
                "This validates realm classification, not coordinate-level "
                "geographic accuracy."
            ),
            (
                "Species with no original-coordinate records cannot contribute "
                "to measured accuracy."
            ),
            (
                "Closely related records within a species are not treated as "
                "independent for threshold tuning; the split and confidence "
                "assessment use species as the unit."
            ),
            (
                "Do not change thresholds from a fallback row labelled "
                "'best_available_but_target_not_confirmed'."
            ),
            "",
        ]
    )
    (output_dir / "gbif_species_range_validation_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("VALIDATION COMPLETED")
    print("=" * 100)
    print(f"Validated species: {total_validated_species:,}")
    print(f"Validated records: {len(records):,}")
    print(
        "Overall species-balanced accuracy: "
        f"{format_number(all_row['species_balanced_accuracy_macro'])}"
    )
    print(f"Current High decision:   {high_decision}")
    print(f"Current Medium decision: {medium_decision}")
    print(f"\nSaved outputs in: {output_dir}")
    for path in sorted(output_dir.glob("*")):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
