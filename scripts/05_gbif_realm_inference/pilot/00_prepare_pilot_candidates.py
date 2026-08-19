#!/usr/bin/env python3
"""Prepare the balanced 200-species candidate sample for the GBIF pilot."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "centroid_results/new_all_seq_with_final_centroids.csv"
DEFAULT_OUTPUT_DIR = "ptp_geographic_recovery_pilot"

REQUIRED_COLUMNS = [
    "species",
    "realm",
    "original_latitude",
    "original_longitude",
    "final_latitude",
    "final_longitude",
    "final_geocode_source",
]


def clean_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(
        result.str.lower().isin({"", "na", "nan", "none", "null"})
    )


def valid_latlon(lat: pd.Series, lon: pd.Series):
    lat_num = pd.to_numeric(lat, errors="coerce")
    lon_num = pd.to_numeric(lon, errors="coerce")

    valid = (
        lat_num.between(-90, 90, inclusive="both")
        & lon_num.between(-180, 180, inclusive="both")
    )
    return valid


def formal_species_mask(species: pd.Series) -> pd.Series:
    return species.str.match(
        r"^[A-Z][A-Za-z.-]+\s+[a-z][A-Za-z.-]+(?:\s|$)",
        na=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the balanced species sample for the GBIF pilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--species-pilot-n",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in header
    ]

    if missing_columns:
        raise SystemExit(f"Required columns missing: {missing_columns}")

    output_dir.mkdir(parents=True, exist_ok=True)

    missing_species_counts = Counter()
    original_species_coordinate_records = Counter()
    original_species_realms = defaultdict(set)

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            input_path,
            usecols=REQUIRED_COLUMNS,
            chunksize=args.chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        for column in ["species", "realm", "final_geocode_source"]:
            chunk[column] = clean_text(chunk[column])

        original_ok = valid_latlon(
            chunk["original_latitude"],
            chunk["original_longitude"],
        )
        final_ok = valid_latlon(
            chunk["final_latitude"],
            chunk["final_longitude"],
        )

        evidence_mask = (
            original_ok
            & chunk["final_geocode_source"].str.lower().eq("original")
        )
        missing_mask = ~final_ok
        species_present = chunk["species"].notna()
        formal_species = formal_species_mask(chunk["species"])

        missing_species_counts.update(
            chunk.loc[
                missing_mask & formal_species,
                "species",
            ].tolist()
        )

        species_evidence = chunk.loc[
            evidence_mask & species_present,
            ["species", "realm"],
        ].copy()

        original_species_coordinate_records.update(
            species_evidence["species"].tolist()
        )

        for species, realms in (
            species_evidence.dropna(subset=["realm"])
            .groupby("species")["realm"]
        ):
            original_species_realms[species].update(
                realms.astype(str).tolist()
            )

        print(f"Processed chunk {chunk_no}: {len(chunk):,} records")

    rows = []

    for species, count in missing_species_counts.items():
        internal_n = original_species_coordinate_records.get(species, 0)

        if internal_n == 0:
            band = "0_internal_locations"
        elif internal_n <= 2:
            band = "1_2_internal_locations"
        elif internal_n <= 9:
            band = "3_9_internal_locations"
        else:
            band = "10plus_internal_locations"

        rows.append(
            {
                "species": species,
                "missing_records_n": count,
                "internal_original_coordinate_records_n": internal_n,
                "internal_original_realms_n": len(
                    original_species_realms.get(species, set())
                ),
                "sampling_band": band,
            }
        )

    species_summary = pd.DataFrame(rows).sort_values(
        "missing_records_n",
        ascending=False,
    )

    summary_path = output_dir / "species_range_candidate_summary.csv"
    species_summary.to_csv(summary_path, index=False)

    if not species_summary.empty:
        bands_n = max(
            1,
            species_summary["sampling_band"].nunique(),
        )
        per_band = math.ceil(args.species_pilot_n / bands_n)
        sampled_bands = []

        for band_no, (_, group) in enumerate(
            species_summary.groupby("sampling_band")
        ):
            sampled_bands.append(
                group.sample(
                    min(len(group), per_band),
                    random_state=args.seed + band_no,
                )
            )

        pilot = pd.concat(
            sampled_bands,
            ignore_index=True,
        ).head(args.species_pilot_n)
    else:
        pilot = species_summary.copy()

    pilot_path = output_dir / "species_range_pilot_candidates.csv"
    pilot.to_csv(pilot_path, index=False)

    print("\nCandidate species:", len(species_summary))
    print("Pilot species:", len(pilot))
    print("Candidate summary:", summary_path)
    print("Pilot sample:", pilot_path)


if __name__ == "__main__":
    main()
