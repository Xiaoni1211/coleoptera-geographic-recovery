#!/usr/bin/env python3

"""
Assign RESOLVE terrestrial biogeographic realms to GBIF occurrence coordinates.

Input:
    gbif_species_range_pilot/gbif_occurrence_pilot/
        gbif_occurrence_records_for_realm.csv

Realm data:
    realm_data/resolve_ecoregions_2017/

Outputs:
    gbif_species_range_pilot/gbif_realm_assignment/
        gbif_occurrence_records_with_realm.csv
        gbif_species_realm_composition.csv
        gbif_species_realm_confidence.csv
        gbif_realm_assignment_unmatched.csv
        gbif_realm_assignment_summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


DEFAULT_INPUT = (
    "gbif_species_range_pilot/"
    "gbif_occurrence_pilot/"
    "gbif_occurrence_records_for_realm.csv"
)

DEFAULT_REALM_DIR = "realm_data/resolve_ecoregions_2017"
DEFAULT_OUTPUT_DIR = (
    "gbif_species_range_pilot/"
    "gbif_realm_assignment"
)


def find_realm_shapefile(realm_dir):
    realm_dir = Path(realm_dir)

    if not realm_dir.exists():
        raise FileNotFoundError(
            f"Realm directory does not exist: {realm_dir}"
        )

    shapefiles = sorted(realm_dir.rglob("*.shp"))

    if not shapefiles:
        raise FileNotFoundError(
            f"No .shp file found under: {realm_dir}"
        )

    preferred = [
        path for path in shapefiles
        if "ecoregion" in path.stem.lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(shapefiles) == 1:
        return shapefiles[0]

    print("\nShapefiles found:")
    for path in shapefiles:
        print(f"  {path}")

    raise RuntimeError(
        "More than one shapefile was found. "
        "Use --realm-file to specify the correct file."
    )


def find_column(columns, candidates, required=True):
    lookup = {str(column).lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise KeyError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
        )

    return None


def first_non_missing(series):
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    values = values.astype(str).str.strip()
    values = values[values.ne("")]

    if len(values) == 0:
        return np.nan

    return values.iloc[0]


def assign_confidence(row):
    """
    Preliminary evidence classification for the pilot.

    This describes the strength and consistency of GBIF realm evidence.
    It is not yet a validated accuracy estimate.
    """

    assigned_locations = row["assigned_evidence_locations_n"]
    dominant_proportion = row["dominant_realm_proportion"]

    if pd.isna(assigned_locations) or assigned_locations == 0:
        return "no_realm_evidence"

    if (
        assigned_locations >= 10
        and dominant_proportion >= 0.90
    ):
        return "high"

    if (
        assigned_locations >= 5
        and dominant_proportion >= 0.70
    ):
        return "medium"

    return "low"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign RESOLVE terrestrial realms to "
            "GBIF occurrence coordinates."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="GBIF occurrence CSV"
    )

    parser.add_argument(
        "--realm-dir",
        default=DEFAULT_REALM_DIR,
        help="Directory containing the RESOLVE shapefile"
    )

    parser.add_argument(
        "--realm-file",
        default=None,
        help="Optional explicit path to the realm shapefile"
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.realm_file:
        realm_file = Path(args.realm_file)
    else:
        realm_file = find_realm_shapefile(args.realm_dir)

    print("=" * 100)
    print("GBIF REALM ASSIGNMENT")
    print("=" * 100)
    print(f"GBIF input:       {input_file}")
    print(f"Realm shapefile:  {realm_file}")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Read GBIF occurrence data
    # ------------------------------------------------------------------

    print("\nReading GBIF occurrence records...")

    records = pd.read_csv(
        input_file,
        low_memory=False
    )

    records["_row_id"] = np.arange(len(records))

    required_columns = [
        "decimalLatitude",
        "decimalLongitude",
        "query_taxonKey",
        "query_scientific_name",
        "input_species_names"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in records.columns
    ]

    if missing_columns:
        raise KeyError(
            "Required GBIF columns are missing: "
            + ", ".join(missing_columns)
        )

    print(f"Records loaded: {len(records):,}")

    # ------------------------------------------------------------------
    # Validate coordinates
    # ------------------------------------------------------------------

    records["decimalLatitude"] = pd.to_numeric(
        records["decimalLatitude"],
        errors="coerce"
    )

    records["decimalLongitude"] = pd.to_numeric(
        records["decimalLongitude"],
        errors="coerce"
    )

    missing_coordinate = (
        records["decimalLatitude"].isna()
        | records["decimalLongitude"].isna()
    )

    out_of_range = (
        ~records["decimalLatitude"].between(-90, 90)
        | ~records["decimalLongitude"].between(-180, 180)
    )

    zero_zero = (
        records["decimalLatitude"].eq(0)
        & records["decimalLongitude"].eq(0)
    )

    records["realm_coordinate_status"] = "valid"

    records.loc[
        missing_coordinate,
        "realm_coordinate_status"
    ] = "missing_or_non_numeric"

    records.loc[
        ~missing_coordinate & out_of_range,
        "realm_coordinate_status"
    ] = "outside_valid_range"

    records.loc[
        ~missing_coordinate & ~out_of_range & zero_zero,
        "realm_coordinate_status"
    ] = "zero_zero"

    valid_mask = records["realm_coordinate_status"].eq("valid")

    print(f"Valid coordinates:   {valid_mask.sum():,}")
    print(f"Invalid coordinates: {(~valid_mask).sum():,}")

    # Coordinate uncertainty is retained as a warning, not used to
    # delete records automatically.
    if "coordinateUncertaintyInMeters" in records.columns:
        uncertainty = pd.to_numeric(
            records["coordinateUncertaintyInMeters"],
            errors="coerce"
        )

        records["coordinate_uncertainty_warning"] = np.where(
            uncertainty > 100000,
            "over_100_km",
            np.where(
                uncertainty.notna(),
                "within_100_km",
                "not_reported"
            )
        )
    else:
        records["coordinate_uncertainty_warning"] = "not_reported"

    # ------------------------------------------------------------------
    # Read RESOLVE ecoregions
    # ------------------------------------------------------------------

    print("\nReading RESOLVE ecoregions...")

    realms = gpd.read_file(realm_file)

    print(f"Polygons loaded: {len(realms):,}")
    print(f"Original CRS:    {realms.crs}")
    print(f"Columns:         {realms.columns.tolist()}")

    realm_col = find_column(
        realms.columns,
        ["REALM", "realm", "REALM_NAME"]
    )

    biome_col = find_column(
        realms.columns,
        ["BIOME_NAME", "biome_name", "BIOME"],
        required=False
    )

    ecoregion_col = find_column(
        realms.columns,
        ["ECO_NAME", "eco_name", "ECOREGION"],
        required=False
    )

    eco_id_col = find_column(
        realms.columns,
        ["ECO_ID", "eco_id"],
        required=False
    )

    selected_columns = [realm_col]

    for column in [biome_col, ecoregion_col, eco_id_col]:
        if column is not None and column not in selected_columns:
            selected_columns.append(column)

    selected_columns.append("geometry")

    realms = realms[selected_columns].copy()

    rename_columns = {
        realm_col: "assigned_realm"
    }

    if biome_col:
        rename_columns[biome_col] = "assigned_biome"

    if ecoregion_col:
        rename_columns[ecoregion_col] = "assigned_ecoregion"

    if eco_id_col:
        rename_columns[eco_id_col] = "assigned_ecoregion_id"

    realms = realms.rename(columns=rename_columns)

    realms = realms[
        realms.geometry.notna()
        & ~realms.geometry.is_empty
    ].copy()

    if realms.crs is None:
        raise ValueError(
            "Realm shapefile has no CRS information."
        )

    realms = realms.to_crs("EPSG:4326")

    print("\nRealm values:")
    print(
        realms["assigned_realm"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Convert valid GBIF coordinates to spatial points
    # ------------------------------------------------------------------

    print("\nCreating GBIF spatial points...")

    valid_records = records.loc[valid_mask].copy()

    points = gpd.GeoDataFrame(
        valid_records,
        geometry=gpd.points_from_xy(
            valid_records["decimalLongitude"],
            valid_records["decimalLatitude"]
        ),
        crs="EPSG:4326"
    )

    # ------------------------------------------------------------------
    # Spatial join
    # ------------------------------------------------------------------

    print("Assigning realms by spatial intersection...")

    joined = gpd.sjoin(
        points,
        realms,
        how="left",
        predicate="intersects"
    )

    # A point exactly on an ecoregion boundary can match more than one
    # polygon. Count matches and realm disagreement explicitly.
    match_stats = (
        joined.groupby("_row_id", as_index=False)
        .agg(
            realm_polygon_matches_n=(
                "assigned_realm",
                "count"
            ),
            distinct_realms_matched_n=(
                "assigned_realm",
                "nunique"
            )
        )
    )

    joined = joined.merge(
        match_stats,
        on="_row_id",
        how="left"
    )

    # Choose one representative row where multiple polygons belong to
    # the same realm. Points matching different realms remain ambiguous.
    joined = (
        joined.sort_values(
            [
                "_row_id",
                "assigned_realm",
                "assigned_ecoregion"
            ]
            if "assigned_ecoregion" in joined.columns
            else ["_row_id", "assigned_realm"],
            na_position="last"
        )
        .drop_duplicates("_row_id", keep="first")
    )

    joined["realm_assignment_status"] = "assigned"

    joined.loc[
        joined["realm_polygon_matches_n"].eq(0),
        "realm_assignment_status"
    ] = "outside_terrestrial_ecoregions"

    joined.loc[
        joined["realm_polygon_matches_n"].gt(1)
        & joined["distinct_realms_matched_n"].eq(1),
        "realm_assignment_status"
    ] = "assigned_multiple_polygons_same_realm"

    ambiguous = joined["distinct_realms_matched_n"].gt(1)

    joined.loc[
        ambiguous,
        "realm_assignment_status"
    ] = "ambiguous_realm_boundary"

    # Do not automatically choose between different realms.
    joined.loc[ambiguous, "assigned_realm"] = np.nan

    assignment_columns = [
        "_row_id",
        "assigned_realm",
        "realm_assignment_status",
        "realm_polygon_matches_n",
        "distinct_realms_matched_n"
    ]

    for column in [
        "assigned_biome",
        "assigned_ecoregion",
        "assigned_ecoregion_id"
    ]:
        if column in joined.columns:
            assignment_columns.append(column)

    assignments = pd.DataFrame(
        joined[assignment_columns]
    )

    records = records.merge(
        assignments,
        on="_row_id",
        how="left"
    )

    records.loc[
        ~valid_mask,
        "realm_assignment_status"
    ] = "invalid_coordinate"

    records["realm_polygon_matches_n"] = (
        records["realm_polygon_matches_n"]
        .fillna(0)
        .astype(int)
    )

    records["distinct_realms_matched_n"] = (
        records["distinct_realms_matched_n"]
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Create spatial evidence units
    # ------------------------------------------------------------------

    # The occurrence download already contains spatial-sampling fields.
    # Use sampling grid cells where available. Otherwise use coordinates
    # rounded to two decimal places as an approximate unique location.
    fallback_location = (
        records["decimalLatitude"].round(2).astype(str)
        + "_"
        + records["decimalLongitude"].round(2).astype(str)
    )

    if "sampling_grid_cell" in records.columns:
        grid_cell = records["sampling_grid_cell"].astype("string")

        records["realm_evidence_location"] = (
            grid_cell.where(
                grid_cell.notna()
                & grid_cell.str.strip().ne(""),
                fallback_location
            )
        )
    else:
        records["realm_evidence_location"] = fallback_location

    assigned = records[
        records["assigned_realm"].notna()
        & records["realm_assignment_status"].isin(
            [
                "assigned",
                "assigned_multiple_polygons_same_realm"
            ]
        )
    ].copy()

    # ------------------------------------------------------------------
    # Realm composition per pilot taxon
    # ------------------------------------------------------------------

    print("\nCalculating species realm composition...")

    composition = (
        assigned.groupby(
            ["query_taxonKey", "assigned_realm"],
            dropna=False,
            as_index=False
        )
        .agg(
            gbif_records_n=("_row_id", "size"),
            evidence_locations_n=(
                "realm_evidence_location",
                "nunique"
            )
        )
    )

    if len(composition) > 0:
        composition["total_assigned_records_n"] = (
            composition.groupby("query_taxonKey")[
                "gbif_records_n"
            ].transform("sum")
        )

        composition["total_evidence_locations_n"] = (
            composition.groupby("query_taxonKey")[
                "evidence_locations_n"
            ].transform("sum")
        )

        composition["record_proportion"] = (
            composition["gbif_records_n"]
            / composition["total_assigned_records_n"]
        )

        composition["evidence_location_proportion"] = (
            composition["evidence_locations_n"]
            / composition["total_evidence_locations_n"]
        )

        composition = composition.sort_values(
            [
                "query_taxonKey",
                "evidence_location_proportion",
                "gbif_records_n"
            ],
            ascending=[True, False, False]
        )

    # ------------------------------------------------------------------
    # Species-level confidence table
    # ------------------------------------------------------------------

    taxon_base = (
        records.groupby(
            "query_taxonKey",
            dropna=False,
            as_index=False
        )
        .agg(
            query_scientific_name=(
                "query_scientific_name",
                first_non_missing
            ),
            input_species_names=(
                "input_species_names",
                first_non_missing
            ),
            input_records_n=("_row_id", "size"),
            valid_coordinate_records_n=(
                "realm_coordinate_status",
                lambda values: values.eq("valid").sum()
            ),
            assigned_realm_records_n=(
                "assigned_realm",
                lambda values: values.notna().sum()
            ),
            unmatched_or_ambiguous_records_n=(
                "realm_assignment_status",
                lambda values: values.isin(
                    [
                        "outside_terrestrial_ecoregions",
                        "ambiguous_realm_boundary",
                        "invalid_coordinate"
                    ]
                ).sum()
            )
        )
    )

    if len(composition) > 0:
        dominant = (
            composition.sort_values(
                [
                    "query_taxonKey",
                    "evidence_location_proportion",
                    "gbif_records_n"
                ],
                ascending=[True, False, False]
            )
            .drop_duplicates("query_taxonKey")
            .rename(
                columns={
                    "assigned_realm": "dominant_realm",
                    "evidence_location_proportion":
                        "dominant_realm_proportion",
                    "total_evidence_locations_n":
                        "assigned_evidence_locations_n"
                }
            )
        )

        realm_count = (
            composition.groupby(
                "query_taxonKey",
                as_index=False
            )["assigned_realm"]
            .nunique()
            .rename(
                columns={
                    "assigned_realm": "realms_detected_n"
                }
            )
        )

        dominant = dominant[
            [
                "query_taxonKey",
                "dominant_realm",
                "dominant_realm_proportion",
                "assigned_evidence_locations_n"
            ]
        ].merge(
            realm_count,
            on="query_taxonKey",
            how="left"
        )

        species_confidence = taxon_base.merge(
            dominant,
            on="query_taxonKey",
            how="left"
        )
    else:
        species_confidence = taxon_base.copy()
        species_confidence["dominant_realm"] = np.nan
        species_confidence["dominant_realm_proportion"] = np.nan
        species_confidence["assigned_evidence_locations_n"] = 0
        species_confidence["realms_detected_n"] = 0

    species_confidence[
        "assigned_evidence_locations_n"
    ] = (
        species_confidence["assigned_evidence_locations_n"]
        .fillna(0)
        .astype(int)
    )

    species_confidence["realms_detected_n"] = (
        species_confidence["realms_detected_n"]
        .fillna(0)
        .astype(int)
    )

    species_confidence["realm_assignment_coverage"] = (
        species_confidence["assigned_realm_records_n"]
        / species_confidence["input_records_n"]
    )

    species_confidence["pilot_realm_confidence"] = (
        species_confidence.apply(
            assign_confidence,
            axis=1
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    status_counts = (
        records["realm_assignment_status"]
        .fillna("not_recorded")
        .value_counts(dropna=False)
    )

    summary_rows = [
        {
            "metric": "input_records",
            "value": len(records)
        },
        {
            "metric": "valid_coordinate_records",
            "value": int(valid_mask.sum())
        },
        {
            "metric": "assigned_realm_records",
            "value": int(records["assigned_realm"].notna().sum())
        },
        {
            "metric": "pilot_taxa",
            "value": int(records["query_taxonKey"].nunique())
        },
        {
            "metric": "taxa_with_realm_evidence",
            "value": int(
                species_confidence["dominant_realm"]
                .notna()
                .sum()
            )
        }
    ]

    for status, count in status_counts.items():
        summary_rows.append(
            {
                "metric": f"status_{status}",
                "value": int(count)
            }
        )

    summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    records = records.drop(columns=["_row_id"])

    unmatched = records[
        records["assigned_realm"].isna()
    ].copy()

    occurrence_output = (
        output_dir
        / "gbif_occurrence_records_with_realm.csv"
    )

    composition_output = (
        output_dir
        / "gbif_species_realm_composition.csv"
    )

    confidence_output = (
        output_dir
        / "gbif_species_realm_confidence.csv"
    )

    unmatched_output = (
        output_dir
        / "gbif_realm_assignment_unmatched.csv"
    )

    summary_output = (
        output_dir
        / "gbif_realm_assignment_summary.csv"
    )

    records.to_csv(occurrence_output, index=False)
    composition.to_csv(composition_output, index=False)
    species_confidence.to_csv(confidence_output, index=False)
    unmatched.to_csv(unmatched_output, index=False)
    summary.to_csv(summary_output, index=False)

    print("\n" + "=" * 100)
    print("REALM ASSIGNMENT COMPLETED")
    print("=" * 100)

    print(f"Input records:          {len(records):,}")
    print(
        "Realm assigned:         "
        f"{records['assigned_realm'].notna().sum():,}"
    )
    print(f"Unassigned records:     {len(unmatched):,}")
    print(
        "Pilot taxa represented: "
        f"{records['query_taxonKey'].nunique():,}"
    )
    print(
        "Taxa with realm result: "
        f"{species_confidence['dominant_realm'].notna().sum():,}"
    )

    print("\nAssignment status:")
    print(status_counts.to_string())

    print("\nSaved files:")
    print(occurrence_output)
    print(composition_output)
    print(confidence_output)
    print(unmatched_output)
    print(summary_output)


if __name__ == "__main__":
    main()
