#!/usr/bin/env python3
"""Prepare full-dataset species-range/GBIF realm-prediction candidates.

This script rebuilds the formal realm-inference target set after applying the
direct coordinate-overlay updates virtually. It then selects records with a
clean binomial species name, attaches any existing PTP result, and writes both
record-level candidates and a unique-species query table.

The master CSV is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_MASTER = Path("centroid_results/new_all_seq_with_final_centroids.csv")
DEFAULT_DIRECT_UPDATES = Path(
    "direct_coordinate_realm_assignment/direct_coordinate_realm_update_table.csv"
)
DEFAULT_PTP_PREDICTIONS = Path(
    "ptp_realm_prediction/ptp_record_realm_predictions_all.csv"
)
DEFAULT_OUTPUT_DIR = Path("gbif_full_prediction")

EXPECTED_MASTER_ROWS = 1_827_573
EXPECTED_DIRECT_UPDATES = 149_427
EXPECTED_FORMAL_INFERENCE_TARGETS = 831_921
EXPECTED_NO_TERRESTRIAL_REALM = 11_856
EXPECTED_POTENTIAL_PTP = 166_462
EXPECTED_PTP_AUTO_UPDATES = 121_654

MISSING_TEXT = {"", "na", "n/a", "nan", "none", "null", "<missing>"}
PLACEHOLDER_TOKENS = {
    "sp",
    "spp",
    "cf",
    "aff",
    "nr",
    "gen",
    "indet",
    "unknown",
    "unidentified",
}
GENUS_RE = re.compile(r"^[A-Z][A-Za-z-]+$")
EPITHET_RE = re.compile(r"^[a-z][A-Za-z-]+$")


def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalise_text_series(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.str.lower().isin(MISSING_TEXT), pd.NA)


def normalise_species(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    if not text or text.lower() in MISSING_TEXT:
        return None
    return text


def classify_species_name(value: object) -> tuple[str | None, str]:
    """Return (normalised name, eligibility flag)."""
    name = normalise_species(value)
    if name is None:
        return None, "missing_species"

    tokens = name.split()
    lowered = [token.lower().rstrip(".") for token in tokens]

    if any(token in PLACEHOLDER_TOKENS for token in lowered):
        return name, "placeholder_or_uncertain_name"
    if len(tokens) != 2:
        return name, "not_two_word_binomial"
    if not GENUS_RE.fullmatch(tokens[0]):
        return name, "invalid_genus_format"
    if not EPITHET_RE.fullmatch(tokens[1]):
        return name, "invalid_specific_epithet_format"
    return name, "clean_binomial"


def parse_boolean_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "t", "1", "yes", "y"})


def load_direct_updates(path: Path) -> pd.DataFrame:
    direct = pd.read_csv(path, low_memory=False)
    require_columns(
        direct,
        {"input_row_number", "BTseq_id", "assigned_realm"},
        "Direct update table",
    )
    direct["input_row_number"] = pd.to_numeric(
        direct["input_row_number"], errors="raise"
    ).astype("int64")
    if direct["input_row_number"].duplicated().any():
        raise ValueError("Direct update table has duplicate input_row_number values")
    if direct["BTseq_id"].isna().any():
        raise ValueError("Direct update table contains missing BTseq_id values")
    if normalise_text_series(direct["assigned_realm"]).isna().any():
        raise ValueError("Direct update table contains missing assigned_realm values")
    return direct.set_index("input_row_number", verify_integrity=True).sort_index()


def load_ptp_predictions(path: Path) -> pd.DataFrame:
    ptp = pd.read_csv(path, low_memory=False)
    required = {
        "input_row_number",
        "BTseq_id",
        "ptp_species",
        "ptp_predicted_realm",
        "ptp_realm_confidence",
        "original_coordinate_locations_n",
        "assigned_realm_locations_n",
        "dominant_realm_locations_n",
        "dominant_realm_proportion",
        "realms_detected_n",
        "realm_overlay_coverage",
        "ptp_auto_assignment_eligible",
    }
    require_columns(ptp, required, "PTP record prediction table")
    ptp["input_row_number"] = pd.to_numeric(
        ptp["input_row_number"], errors="raise"
    ).astype("int64")
    if ptp["input_row_number"].duplicated().any():
        raise ValueError("PTP record table has duplicate input_row_number values")
    if ptp["BTseq_id"].isna().any():
        raise ValueError("PTP record table contains missing BTseq_id values")
    return ptp.set_index("input_row_number", verify_integrity=True).sort_index()


def build_species_table(
    species_total: Counter,
    species_ptp_high: Counter,
    species_ptp_medium: Counter,
    species_ptp_low: Counter,
    species_ptp_insufficient: Counter,
    species_ptp_none: Counter,
    species_ptp_auto: Counter,
) -> pd.DataFrame:
    rows = []
    for species in sorted(species_total):
        rows.append(
            {
                "input_species_names": species,
                "candidate_records_n": species_total[species],
                "ptp_high_records_n": species_ptp_high[species],
                "ptp_medium_records_n": species_ptp_medium[species],
                "ptp_low_records_n": species_ptp_low[species],
                "ptp_insufficient_records_n": species_ptp_insufficient[species],
                "no_ptp_candidate_records_n": species_ptp_none[species],
                "ptp_auto_assignment_records_n": species_ptp_auto[species],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full GBIF species-range realm-prediction candidates."
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--direct-updates", type=Path, default=DEFAULT_DIRECT_UPDATES)
    parser.add_argument("--ptp-predictions", type=Path, default=DEFAULT_PTP_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--skip-expected-count-checks",
        action="store_true",
        help="Allow use with a deliberately changed master dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.master, args.direct_updates, args.ptp_predictions):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "gbif_full_candidate_records.csv"
    species_path = args.output_dir / "gbif_full_candidate_species.csv"
    exclusions_path = args.output_dir / "gbif_full_species_name_exclusions.csv"
    inventory_path = args.output_dir / "gbif_full_candidate_inventory.csv"
    config_path = args.output_dir / "gbif_full_candidate_config.json"

    banner("PREPARE FULL SPECIES-RANGE / GBIF CANDIDATES")
    print(f"Master:          {args.master}")
    print(f"Direct updates:  {args.direct_updates}")
    print(f"PTP predictions: {args.ptp_predictions}")
    print(f"Output:          {args.output_dir}")
    print(f"Chunk size:      {args.chunksize:,}")

    banner("LOAD DIRECT AND PTP TABLES")
    direct = load_direct_updates(args.direct_updates)
    ptp = load_ptp_predictions(args.ptp_predictions)
    print(f"Direct update records: {len(direct):,}")
    print(f"PTP candidate records: {len(ptp):,}")
    print(
        "PTP automatic updates: "
        f"{parse_boolean_series(ptp['ptp_auto_assignment_eligible']).sum():,}"
    )

    direct_seen: set[int] = set()
    ptp_seen: set[int] = set()
    master_rows = 0
    original_realm_missing_n = 0
    direct_updates_applied_n = 0
    direct_existing_realm_preserved_n = 0
    effective_realm_missing_n = 0
    coordinate_flag_differences_n = 0
    no_terrestrial_realm_n = 0
    formal_inference_targets_n = 0
    targets_with_species_value_n = 0
    clean_candidate_records_n = 0
    excluded_species_records_n = 0
    ptp_candidates_among_targets_n = 0
    ptp_auto_among_targets_n = 0
    ptp_candidates_among_gbif_candidates_n = 0
    ptp_auto_among_gbif_candidates_n = 0

    species_total: Counter = Counter()
    species_ptp_high: Counter = Counter()
    species_ptp_medium: Counter = Counter()
    species_ptp_low: Counter = Counter()
    species_ptp_insufficient: Counter = Counter()
    species_ptp_none: Counter = Counter()
    species_ptp_auto: Counter = Counter()
    exclusions: Counter = Counter()

    output_header_written = False
    if records_path.exists():
        records_path.unlink()

    master_required = {
        "BTseq_id",
        "db_id",
        "species",
        "ptp_species",
        "realm",
        "final_latitude",
        "final_longitude",
        "final_coordinate_valid",
    }

    banner("SCAN MASTER AND BUILD CANDIDATES")
    for chunk_number, chunk in enumerate(
        pd.read_csv(args.master, chunksize=args.chunksize, low_memory=False), start=1
    ):
        require_columns(chunk, master_required, "Master CSV")
        n = len(chunk)
        start = master_rows
        row_numbers = pd.RangeIndex(start, start + n, name="input_row_number")
        chunk = chunk.reset_index(drop=True)
        chunk.insert(0, "input_row_number", row_numbers.to_numpy())

        original_realm = normalise_text_series(chunk["realm"])
        original_missing = original_realm.isna()
        original_realm_missing_n += int(original_missing.sum())

        direct_slice = direct.reindex(row_numbers)
        direct_mask = direct_slice["assigned_realm"].notna().to_numpy()
        if direct_mask.any():
            positions = direct_mask.nonzero()[0]
            expected_btseq = direct_slice.iloc[positions]["BTseq_id"].astype(str).to_numpy()
            observed_btseq = chunk.iloc[positions]["BTseq_id"].astype(str).to_numpy()
            mismatch = expected_btseq != observed_btseq
            if mismatch.any():
                bad_pos = positions[mismatch][0]
                bad_row = start + int(bad_pos)
                raise ValueError(
                    "Direct update identity mismatch at input_row_number "
                    f"{bad_row}: master={chunk.iloc[bad_pos]['BTseq_id']!r}, "
                    f"update={direct.loc[bad_row, 'BTseq_id']!r}"
                )
            direct_seen.update(int(row_numbers[pos]) for pos in positions)

        effective_realm = original_realm.copy()
        can_apply_direct = direct_mask & original_missing.to_numpy()
        preserve_existing = direct_mask & ~original_missing.to_numpy()
        if can_apply_direct.any():
            effective_realm.iloc[can_apply_direct.nonzero()[0]] = (
                direct_slice.loc[can_apply_direct, "assigned_realm"].astype("string").to_numpy()
            )
        direct_updates_applied_n += int(can_apply_direct.sum())
        direct_existing_realm_preserved_n += int(preserve_existing.sum())

        lat = pd.to_numeric(chunk["final_latitude"], errors="coerce")
        lon = pd.to_numeric(chunk["final_longitude"], errors="coerce")
        coordinate_valid = (
            lat.notna()
            & lon.notna()
            & lat.between(-90, 90, inclusive="both")
            & lon.between(-180, 180, inclusive="both")
            & ~((lat == 0) & (lon == 0))
        )
        stored_coordinate_valid = parse_boolean_series(chunk["final_coordinate_valid"])
        coordinate_flag_differences_n += int(
            (coordinate_valid != stored_coordinate_valid).sum()
        )

        effective_missing = effective_realm.isna()
        effective_realm_missing_n += int(effective_missing.sum())
        no_terrestrial_mask = effective_missing & coordinate_valid
        formal_target_mask = effective_missing & ~coordinate_valid
        no_terrestrial_realm_n += int(no_terrestrial_mask.sum())
        formal_inference_targets_n += int(formal_target_mask.sum())

        ptp_slice = ptp.reindex(row_numbers)
        ptp_mask = ptp_slice["BTseq_id"].notna().to_numpy()
        if ptp_mask.any():
            positions = ptp_mask.nonzero()[0]
            expected_btseq = ptp_slice.iloc[positions]["BTseq_id"].astype(str).to_numpy()
            observed_btseq = chunk.iloc[positions]["BTseq_id"].astype(str).to_numpy()
            mismatch = expected_btseq != observed_btseq
            if mismatch.any():
                bad_pos = positions[mismatch][0]
                bad_row = start + int(bad_pos)
                raise ValueError(
                    "PTP identity mismatch at input_row_number "
                    f"{bad_row}: master={chunk.iloc[bad_pos]['BTseq_id']!r}, "
                    f"PTP={ptp.loc[bad_row, 'BTseq_id']!r}"
                )
            ptp_seen.update(int(row_numbers[pos]) for pos in positions)

        ptp_auto = pd.Series(False, index=chunk.index)
        if ptp_mask.any():
            ptp_auto.iloc[ptp_mask.nonzero()[0]] = parse_boolean_series(
                ptp_slice.loc[ptp_mask, "ptp_auto_assignment_eligible"]
            ).to_numpy()
        ptp_candidates_among_targets_n += int(
            (formal_target_mask.to_numpy() & ptp_mask).sum()
        )
        ptp_auto_among_targets_n += int((formal_target_mask & ptp_auto).sum())

        target_positions = formal_target_mask[formal_target_mask].index
        if len(target_positions):
            classified = chunk.loc[target_positions, "species"].map(classify_species_name)
            normalised = classified.map(lambda item: item[0])
            flags = classified.map(lambda item: item[1])
            species_present = flags != "missing_species"
            eligible = flags == "clean_binomial"
            targets_with_species_value_n += int(species_present.sum())
            clean_candidate_records_n += int(eligible.sum())
            excluded_species_records_n += int((species_present & ~eligible).sum())

            for name, flag in zip(normalised[species_present & ~eligible], flags[species_present & ~eligible]):
                exclusions[(name, flag)] += 1

            candidate_positions = target_positions[eligible.to_numpy()]
            if len(candidate_positions):
                out = chunk.loc[candidate_positions].copy()
                out["species_normalized"] = normalised.loc[candidate_positions].to_numpy()
                out["species_name_flag"] = "clean_binomial"
                out["computed_final_coordinate_valid"] = False
                out["formal_inference_target"] = True

                ptp_columns = [
                    "ptp_predicted_realm",
                    "ptp_realm_confidence",
                    "original_coordinate_locations_n",
                    "assigned_realm_locations_n",
                    "dominant_realm_locations_n",
                    "dominant_realm_proportion",
                    "realms_detected_n",
                    "realm_overlay_coverage",
                    "ptp_auto_assignment_eligible",
                ]
                aligned_ptp = ptp_slice.reindex(out["input_row_number"])
                for col in ptp_columns:
                    out[col] = aligned_ptp[col].to_numpy()
                out["ptp_candidate"] = aligned_ptp["BTseq_id"].notna().to_numpy()
                out["ptp_auto_assignment_eligible"] = parse_boolean_series(
                    out["ptp_auto_assignment_eligible"]
                )

                keep_columns = [
                    "input_row_number",
                    "BTseq_id",
                    "db_id",
                    "species",
                    "species_normalized",
                    "species_name_flag",
                    "ptp_species",
                    "country",
                    "province_state",
                    "region",
                    "sector",
                    "site",
                    "realm",
                    "final_latitude",
                    "final_longitude",
                    "final_coordinate_valid",
                    "computed_final_coordinate_valid",
                    "formal_inference_target",
                    "ptp_candidate",
                    "ptp_predicted_realm",
                    "ptp_realm_confidence",
                    "original_coordinate_locations_n",
                    "assigned_realm_locations_n",
                    "dominant_realm_locations_n",
                    "dominant_realm_proportion",
                    "realms_detected_n",
                    "realm_overlay_coverage",
                    "ptp_auto_assignment_eligible",
                ]
                keep_columns = [col for col in keep_columns if col in out.columns]
                out = out[keep_columns]
                out.to_csv(
                    records_path,
                    mode="a",
                    header=not output_header_written,
                    index=False,
                )
                output_header_written = True

                names = out["species_normalized"].astype(str)
                confidences = out["ptp_realm_confidence"].fillna("No PTP")
                autos = out["ptp_auto_assignment_eligible"].fillna(False).astype(bool)
                species_total.update(names)
                species_ptp_high.update(names[confidences.eq("High")])
                species_ptp_medium.update(names[confidences.eq("Medium")])
                species_ptp_low.update(names[confidences.eq("Low")])
                species_ptp_insufficient.update(names[confidences.eq("Insufficient")])
                species_ptp_none.update(names[confidences.eq("No PTP")])
                species_ptp_auto.update(names[autos])
                ptp_candidates_among_gbif_candidates_n += int(out["ptp_candidate"].sum())
                ptp_auto_among_gbif_candidates_n += int(autos.sum())

        master_rows += n
        print(
            f"Chunk {chunk_number}: rows={n:,} | "
            f"formal targets={int(formal_target_mask.sum()):,} | "
            f"clean species candidates={int((flags == 'clean_binomial').sum()) if len(target_positions) else 0:,} | "
            f"cumulative rows={master_rows:,}"
        )

    if not output_header_written:
        raise RuntimeError("No clean-binomial GBIF candidate records were produced")

    unseen_direct = sorted(set(direct.index).difference(direct_seen))
    unseen_ptp = sorted(set(ptp.index).difference(ptp_seen))
    if unseen_direct:
        raise ValueError(
            f"{len(unseen_direct):,} direct updates did not match the master; "
            f"first input_row_number={unseen_direct[0]}"
        )
    if unseen_ptp:
        raise ValueError(
            f"{len(unseen_ptp):,} PTP predictions did not match the master; "
            f"first input_row_number={unseen_ptp[0]}"
        )

    species_df = build_species_table(
        species_total,
        species_ptp_high,
        species_ptp_medium,
        species_ptp_low,
        species_ptp_insufficient,
        species_ptp_none,
        species_ptp_auto,
    )
    species_df.to_csv(species_path, index=False)

    exclusions_df = pd.DataFrame(
        [
            {"species_name": name, "exclusion_reason": reason, "records_n": count}
            for (name, reason), count in sorted(
                exclusions.items(), key=lambda item: (-item[1], str(item[0][0]))
            )
        ]
    )
    exclusions_df.to_csv(exclusions_path, index=False)

    inventory = {
        "master_rows": master_rows,
        "original_realm_missing_records": original_realm_missing_n,
        "direct_update_records": len(direct),
        "direct_updates_applied": direct_updates_applied_n,
        "direct_updates_preserved_existing_realm": direct_existing_realm_preserved_n,
        "effective_realm_missing_after_direct_updates": effective_realm_missing_n,
        "coordinate_flag_differences": coordinate_flag_differences_n,
        "no_terrestrial_realm_records": no_terrestrial_realm_n,
        "formal_inference_targets": formal_inference_targets_n,
        "formal_targets_with_species_value": targets_with_species_value_n,
        "gbif_clean_binomial_candidate_records": clean_candidate_records_n,
        "species_value_records_excluded_from_gbif": excluded_species_records_n,
        "unique_gbif_candidate_species": len(species_df),
        "ptp_candidate_records_among_formal_targets": ptp_candidates_among_targets_n,
        "ptp_auto_records_among_formal_targets": ptp_auto_among_targets_n,
        "ptp_candidate_records_among_gbif_candidates": ptp_candidates_among_gbif_candidates_n,
        "ptp_auto_records_among_gbif_candidates": ptp_auto_among_gbif_candidates_n,
    }
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in inventory.items()]
    ).to_csv(inventory_path, index=False)

    config = {
        "master": str(args.master),
        "direct_updates": str(args.direct_updates),
        "ptp_predictions": str(args.ptp_predictions),
        "output_dir": str(args.output_dir),
        "chunksize": args.chunksize,
        "input_row_number_definition": "physical_zero_based_master_csv_data_row",
        "identity_check": "input_row_number_plus_BTseq_id",
        "coordinate_validity_definition": (
            "numeric final latitude/longitude in valid bounds, excluding exactly 0,0"
        ),
        "formal_inference_target_definition": (
            "effective realm missing after virtual direct update AND no valid final coordinate"
        ),
        "species_candidate_definition": "formal inference target with clean two-word binomial",
        "observed": inventory,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    expected_checks = {
        "master_rows": (master_rows, EXPECTED_MASTER_ROWS),
        "direct_updates": (len(direct), EXPECTED_DIRECT_UPDATES),
        "formal_inference_targets": (
            formal_inference_targets_n,
            EXPECTED_FORMAL_INFERENCE_TARGETS,
        ),
        "no_terrestrial_realm_records": (
            no_terrestrial_realm_n,
            EXPECTED_NO_TERRESTRIAL_REALM,
        ),
        "PTP candidate records": (len(ptp), EXPECTED_POTENTIAL_PTP),
        "PTP automatic updates": (
            int(parse_boolean_series(ptp["ptp_auto_assignment_eligible"]).sum()),
            EXPECTED_PTP_AUTO_UPDATES,
        ),
        "PTP candidates among formal targets": (
            ptp_candidates_among_targets_n,
            EXPECTED_POTENTIAL_PTP,
        ),
        "PTP auto records among formal targets": (
            ptp_auto_among_targets_n,
            EXPECTED_PTP_AUTO_UPDATES,
        ),
    }
    mismatches = [
        f"{label}: observed {observed:,}, expected {expected:,}"
        for label, (observed, expected) in expected_checks.items()
        if observed != expected
    ]
    if mismatches and not args.skip_expected_count_checks:
        raise ValueError("Expected-count validation failed:\n" + "\n".join(mismatches))

    banner("FINAL SUMMARY")
    for key, value in inventory.items():
        print(f"{key:52s} {value:>12,}")
    if mismatches:
        print("\nWARNING: expected-count mismatches were explicitly allowed:")
        for message in mismatches:
            print(f"  {message}")

    banner("SAVED FILES")
    for path in (
        records_path,
        species_path,
        exclusions_path,
        inventory_path,
        config_path,
    ):
        print(path)
    print("\nFull GBIF candidate preparation completed successfully.")


if __name__ == "__main__":
    main()
