#!/usr/bin/env python3
"""
Check and standardise unique province/country centroid queries before matching
them to Natural Earth.

This script DOES NOT calculate centroids and DOES NOT overwrite original
columns. It creates conservative matching keys and diagnostic files.

Default inputs
--------------
unique_province_centroid_queries.csv
unique_country_centroid_queries.csv
province_without_country_identifier.csv  (optional)

Main outputs (in centroid_standardisation/)
------------------------------------------
standardised_unique_province_centroid_queries.csv
standardised_unique_country_centroid_queries.csv
province_standardisation_issues.csv
country_standardisation_issues.csv
duplicate_standardised_province_keys.csv
duplicate_standardised_country_keys.csv
province_without_country_identifier_checked.csv  (if input exists)
centroid_query_standardisation_summary.csv

Requirement: pandas and geopandas (the same packages needed for the centroid step)
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise SystemExit(
        "Missing package 'geopandas'. Install it with:\n"
        "  conda install -c conda-forge geopandas"
    ) from exc


MISSING_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "unknown", "unk", "missing",
    "not known", "not specified", "no data", "no locality", "-", "--", "?",
}

# Only exact aliases are accepted automatically. Fuzzy country matching is
# deliberately avoided at this stage.
COUNTRY_ALIASES = {
    "bolivia": "BOL",
    "brunei": "BRN",
    "cape verde": "CPV",
    "cabo verde": "CPV",
    "congo": "COG",
    "republic of congo": "COG",
    "congo brazzaville": "COG",
    "democratic republic of congo": "COD",
    "democratic republic of the congo": "COD",
    "dr congo": "COD",
    "drc": "COD",
    "congo kinshasa": "COD",
    "cote d ivoire": "CIV",
    "ivory coast": "CIV",
    "czech republic": "CZE",
    "czechia": "CZE",
    "east timor": "TLS",
    "timor leste": "TLS",
    "iran": "IRN",
    "laos": "LAO",
    "moldova": "MDA",
    "north korea": "PRK",
    "south korea": "KOR",
    "korea north": "PRK",
    "korea south": "KOR",
    "palestine": "PSE",
    "russia": "RUS",
    "russian federation": "RUS",
    "syria": "SYR",
    "taiwan": "TWN",
    "tanzania": "TZA",
    "the bahamas": "BHS",
    "bahamas": "BHS",
    "the gambia": "GMB",
    "gambia": "GMB",
    "turkey": "TUR",
    "turkiye": "TUR",
    "uk": "GBR",
    "u k": "GBR",
    "great britain": "GBR",
    "united kingdom": "GBR",
    "usa": "USA",
    "u s a": "USA",
    "us": "USA",
    "u s": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "venezuela": "VEN",
    "vietnam": "VNM",
    "viet nam": "VNM",
    "micronesia": "FSM",
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "macedonia": "MKD",
    "north macedonia": "MKD",
    "burma": "MMR",
    "myanmar": "MMR",
    "sao tome and principe": "STP",
    "french guiana": "GUF",
    "guadeloupe": "GLP",
    "reunion": "REU",
    "la reunion": "REU",
    "falkland islands": "FLK",
    "falkland islands islas malvinas": "FLK",
    "kosovo": "KOS",
}

# Common ISO alpha-2 codes that Natural Earth may store as -99 even though its
# ADM0_A3 field contains a usable territory/admin key.
CODE_ALIASES = {
    "GF": "GUF",
    "GP": "GLP",
    "RE": "REU",
    "XK": "KOS",
}

# These are detected but are NOT automatically converted to current countries.
# Historical or non-sovereign labels require a documented decision.
REVIEW_COUNTRY_TERMS = {
    "ussr", "soviet union", "yugoslavia", "czechoslovakia", "zaire",
    "netherlands antilles", "serbia and montenegro", "west germany",
    "east germany", "antarctica",
}

COUNTRY_COL_ALIASES = ["country", "country_name", "nation"]
ISO_COL_ALIASES = [
    "country_iso", "country_code", "countrycode", "iso", "iso2", "iso3",
    "iso_a2", "iso_a3", "adm0_a3",
]
PROVINCE_COL_ALIASES = [
    "province_state", "province", "state", "state_province", "admin1",
    "first_order_administrative_division",
]


def clean_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return "" if text.casefold() in MISSING_TOKENS else text


def ascii_key(value: object) -> str:
    text = clean_cell(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def province_relaxed_key(value: object) -> str:
    """A secondary key only; never used to overwrite the original name."""
    key = ascii_key(value)
    if not key:
        return ""
    suffixes = (
        " province", " state", " region", " territory", " governorate",
        " prefecture", " department", " county", " district", " oblast",
        " autonomous region", " administrative region",
    )
    prefixes = ("province of ", "state of ", "region of ", "department of ")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix) and len(key) > len(prefix):
                key = key[len(prefix):].strip()
                changed = True
        for suffix in suffixes:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[:-len(suffix)].strip()
                changed = True
    return key


def normalise_column_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")


def find_column(df: pd.DataFrame, aliases: list[str], required: bool) -> str | None:
    lookup = {normalise_column_token(col): col for col in df.columns}
    for alias in aliases:
        token = normalise_column_token(alias)
        if token in lookup:
            return lookup[token]
    if required:
        raise ValueError(
            f"Could not find a required column. Tried {aliases}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


class CountryReference:
    """Exact country-name/code reference built from Natural Earth Admin-0."""

    def __init__(self, admin0_path: Path):
        if not admin0_path.exists():
            raise FileNotFoundError(f"Natural Earth Admin-0 file not found: {admin0_path}")
        admin0 = gpd.read_file(admin0_path, ignore_geometry=True)
        if admin0.empty:
            raise ValueError(f"Natural Earth Admin-0 contains no rows: {admin0_path}")

        columns = {str(c).upper(): c for c in admin0.columns}
        self.code3_cols = [
            columns[x] for x in
            # GU_A3/SU_A3 come first so overseas territories retain their own
            # geographic-unit key instead of inheriting the sovereign ADM0 key.
            ["GU_A3", "SU_A3", "ADM0_A3", "ISO_A3", "WB_A3", "BRK_A3"]
            if x in columns
        ]
        self.code2_cols = [columns[x] for x in ["ISO_A2", "WB_A2"] if x in columns]
        self.numeric_cols = [columns[x] for x in ["ISO_N3", "UN_A3"] if x in columns]
        self.name_cols = [
            columns[x] for x in
            [
                "ADMIN", "NAME", "NAME_LONG", "NAME_EN", "SOVEREIGNT",
                "GEOUNIT", "SUBUNIT", "BRK_NAME", "FORMAL_EN", "ABBREV",
                "POSTAL",
            ]
            if x in columns
        ]
        # SOVEREIGNT is deliberately not used as a country-name alias. For
        # example, several overseas territories have SOVEREIGNT='United
        # Kingdom'; treating it as an alias makes the real UK falsely
        # ambiguous. Keep ADMIN/NAME/NAME_LONG as the target entity names.
        sovereign_col = columns.get("SOVEREIGNT")
        if sovereign_col in self.name_cols:
            self.name_cols.remove(sovereign_col)
        if not self.code3_cols or not self.name_cols:
            raise ValueError(
                "Natural Earth Admin-0 lacks expected code/name fields. "
                f"Available fields: {list(admin0.columns)}"
            )

        self.code_lookup: dict[str, set[str]] = {}
        self.name_lookup: dict[str, set[str]] = {}
        self.canonical_name: dict[str, str] = {}

        for _, row in admin0.iterrows():
            codes3 = [
                clean_cell(row[col]).upper() for col in self.code3_cols
                if clean_cell(row[col]) not in {"", "-99"}
            ]
            if not codes3:
                continue
            # ADM0_A3 is first when available and is Natural Earth's stable key.
            standard_code = codes3[0]
            codes2 = [
                clean_cell(row[col]).upper() for col in self.code2_cols
                if clean_cell(row[col]) not in {"", "-99"}
            ]
            numeric = [
                clean_cell(row[col]).zfill(3) for col in self.numeric_cols
                if clean_cell(row[col]) not in {"", "-99"}
            ]
            names = [
                clean_cell(row[col]) for col in self.name_cols if clean_cell(row[col])
            ]
            adm0_col = columns.get("ADM0_A3")
            adm0_code = clean_cell(row[adm0_col]).upper() if adm0_col else ""
            if adm0_code not in {"", "-99"} and standard_code != adm0_code:
                # A territory row may still contain ADMIN/NAME='France' or
                # another sovereign label. Do not let those parent labels make
                # the sovereign country ambiguous; retain its GEOUNIT/SUBUNIT
                # names instead.
                parent_keys = {
                    ascii_key(row[col])
                    for field in ["ADMIN", "SOVEREIGNT", "FORMAL_EN"]
                    for col in [columns.get(field)]
                    if col is not None and clean_cell(row[col])
                }
                names = [name for name in names if ascii_key(name) not in parent_keys]

            for code in set(codes3 + codes2 + numeric):
                self.code_lookup.setdefault(code, set()).add(standard_code)
            for name in names:
                self.name_lookup.setdefault(ascii_key(name), set()).add(standard_code)
            if standard_code not in self.canonical_name:
                self.canonical_name[standard_code] = names[0] if names else standard_code

        # Add only explicit, deterministic aliases whose target exists in Admin-0.
        for alias, target in COUNTRY_ALIASES.items():
            targets = self.code_lookup.get(target, set())
            if len(targets) == 1:
                self.name_lookup.setdefault(alias, set()).update(targets)
        for alias, target in CODE_ALIASES.items():
            targets = self.code_lookup.get(target, set())
            if len(targets) == 1:
                self.code_lookup.setdefault(alias, set()).update(targets)

    def resolve_code(self, value: object) -> tuple[str, str]:
        raw = re.sub(r"[^A-Za-z0-9]", "", clean_cell(value)).upper()
        if not raw:
            return "", "missing"
        if raw.isdigit():
            raw = raw.zfill(3)
        matches = self.code_lookup.get(raw, set())
        if len(matches) == 1:
            return next(iter(matches)), "valid"
        if len(matches) > 1:
            return "", "ambiguous"
        return "", "invalid"

    def resolve_name(self, value: object) -> tuple[str, str, str]:
        key = ascii_key(value)
        if not key:
            return "", "missing", ""
        if key in REVIEW_COUNTRY_TERMS:
            return "", "needs_manual_review", "historical_or_special_name"
        matches = self.name_lookup.get(key, set())
        if len(matches) == 1:
            method = "exact_alias" if key in COUNTRY_ALIASES else "natural_earth_exact_name"
            return next(iter(matches)), "resolved", method
        if len(matches) > 1:
            return "", "ambiguous", "multiple_natural_earth_name_matches"

        # Some source rows contain country plus lower-level locality text in the
        # country field, e.g. 'United Kingdom:England|Wimbledon Common'. Accept
        # only an exact recognised prefix before ':'; never fuzzy-match it.
        raw = clean_cell(value)
        if ":" in raw:
            prefix_key = ascii_key(raw.split(":", 1)[0])
            prefix_matches = self.name_lookup.get(prefix_key, set())
            if len(prefix_matches) == 1:
                return (
                    next(iter(prefix_matches)),
                    "resolved",
                    "exact_country_prefix_before_locality",
                )
        return "", "unresolved", "country_name_not_recognised"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")


def country_diagnostics(
    country: object, iso: object, reference: CountryReference
) -> dict[str, str]:
    country_original = clean_cell(country)
    iso_original = clean_cell(iso)
    code_iso3, code_status = reference.resolve_code(iso_original)
    name_iso3, name_status, name_method = reference.resolve_name(country_original)

    conflict = bool(code_iso3 and name_iso3 and code_iso3 != name_iso3)
    if conflict:
        final_iso3 = ""
        status = "conflict_needs_manual_review"
        method = "valid_iso_conflicts_with_country_name"
    elif code_iso3:
        final_iso3 = code_iso3
        status = "resolved"
        method = "valid_input_iso"
    elif name_iso3:
        final_iso3 = name_iso3
        status = "resolved"
        method = name_method
    elif not country_original and not iso_original:
        final_iso3 = ""
        status = "missing_country_identifier"
        method = "none"
    elif name_status == "needs_manual_review":
        final_iso3 = ""
        status = "country_name_needs_manual_review"
        method = name_method
    else:
        final_iso3 = ""
        status = "unresolved_country"
        if code_status == "invalid" and name_status in {"unresolved", "missing"}:
            method = "invalid_iso_and_unresolved_name"
        elif code_status == "invalid":
            method = "invalid_input_iso"
        else:
            method = name_method or "country_not_resolved"

    canonical = reference.canonical_name.get(final_iso3, "")

    return {
        "country_original_checked": country_original,
        "country_normalised_key": ascii_key(country_original),
        "country_iso_original_checked": iso_original,
        "country_iso_input_status": code_status,
        "country_iso3_from_input": code_iso3,
        "country_iso3_from_name": name_iso3,
        "country_name_resolution_status": name_status,
        "country_name_resolution_method": name_method,
        "country_iso_name_conflict": str(conflict),
        "country_standardisation_status": status,
        "country_standardisation_method": method,
        "country_standardised": canonical,
        "country_iso3_standardised": final_iso3,
    }


def standardise_country_fields(
    df: pd.DataFrame,
    country_col: str | None,
    iso_col: str | None,
    reference: CountryReference,
) -> pd.DataFrame:
    countries = df[country_col] if country_col else pd.Series("", index=df.index)
    codes = df[iso_col] if iso_col else pd.Series("", index=df.index)
    details = pd.DataFrame(
        [
            country_diagnostics(country, iso, reference)
            for country, iso in zip(countries, codes)
        ],
        index=df.index,
    )
    return pd.concat([df.copy(), details], axis=1)


def add_province_fields(df: pd.DataFrame, province_col: str) -> pd.DataFrame:
    out = df.copy()
    out["province_state_original_checked"] = out[province_col].map(clean_cell)
    out["province_state_normalised_key"] = out[province_col].map(ascii_key)
    out["province_state_relaxed_key"] = out[province_col].map(province_relaxed_key)
    out["province_state_missing"] = out["province_state_original_checked"].eq("")
    # Short alphabetic values are flagged, not automatically expanded.
    out["province_state_possible_abbreviation"] = out[
        "province_state_original_checked"
    ].map(lambda x: bool(re.fullmatch(r"[A-Za-z]{1,3}", x.replace(" ", ""))))
    out["province_match_key"] = (
        out["country_iso3_standardised"].fillna("")
        + "|"
        + out["province_state_normalised_key"].fillna("")
    )
    out["province_relaxed_match_key"] = (
        out["country_iso3_standardised"].fillna("")
        + "|"
        + out["province_state_relaxed_key"].fillna("")
    )
    out["province_standardisation_status"] = "ready_for_natural_earth_matching"
    out.loc[
        out["country_standardisation_status"].ne("resolved"),
        "province_standardisation_status",
    ] = "country_not_resolved"
    out.loc[out["province_state_missing"], "province_standardisation_status"] = (
        "province_missing"
    )
    out.loc[
        out["province_state_possible_abbreviation"]
        & out["province_standardisation_status"].eq("ready_for_natural_earth_matching"),
        "province_standardisation_status",
    ] = "ready_but_abbreviation_needs_attention"
    return out


def duplicate_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    valid = df[key].fillna("").ne("") & ~df[key].str.endswith("|")
    dup = valid & df.duplicated(key, keep=False)
    return df.loc[dup].sort_values(key).copy()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def summary_row(dataset: str, metric: str, value: int) -> dict[str, object]:
    return {"dataset": dataset, "metric": metric, "value": int(value)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check and standardise centroid query combinations."
    )
    parser.add_argument(
        "--admin0",
        default="natural_earth/admin0/ne_10m_admin_0_countries.shp",
        help="Natural Earth Admin-0 shapefile used as the country reference."
    )
    parser.add_argument(
        "--province", default="unique_province_centroid_queries.csv",
        help="Unique province centroid query CSV."
    )
    parser.add_argument(
        "--country", default="unique_country_centroid_queries.csv",
        help="Unique country centroid query CSV."
    )
    parser.add_argument(
        "--province-without-country",
        default="province_without_country_identifier.csv",
        help="Optional province-without-country CSV."
    )
    parser.add_argument(
        "--output-dir", default="centroid_standardisation",
        help="Directory for output CSV files."
    )
    args = parser.parse_args()

    province_path = Path(args.province)
    country_path = Path(args.country)
    no_country_path = Path(args.province_without_country)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)

    reference = CountryReference(Path(args.admin0))
    print(f"Natural Earth country keys loaded: {len(reference.canonical_name):,}")
    print("CENTROID QUERY CHECKING AND STANDARDISATION")
    print("=" * 80)

    province = read_csv(province_path)
    country = read_csv(country_path)
    print(f"Province combinations loaded: {len(province):,}")
    print(f"Country combinations loaded:  {len(country):,}")

    p_country_col = find_column(province, COUNTRY_COL_ALIASES, required=False)
    p_iso_col = find_column(province, ISO_COL_ALIASES, required=False)
    p_province_col = find_column(province, PROVINCE_COL_ALIASES, required=True)
    if p_country_col is None and p_iso_col is None:
        raise ValueError(
            "Province query file contains neither a country column nor an ISO column."
        )

    c_country_col = find_column(country, COUNTRY_COL_ALIASES, required=False)
    c_iso_col = find_column(country, ISO_COL_ALIASES, required=False)
    if c_country_col is None and c_iso_col is None:
        raise ValueError(
            "Country query file contains neither a country column nor an ISO column."
        )

    province_std = standardise_country_fields(
        province, p_country_col, p_iso_col, reference
    )
    province_std = add_province_fields(province_std, p_province_col)
    country_std = standardise_country_fields(
        country, c_country_col, c_iso_col, reference
    )
    country_std["country_match_key"] = country_std["country_iso3_standardised"]
    country_std["country_ready_for_natural_earth_matching"] = country_std[
        "country_standardisation_status"
    ].eq("resolved")

    province_issues = province_std.loc[
        province_std["province_standardisation_status"].ne(
            "ready_for_natural_earth_matching"
        )
    ].copy()
    country_issues = country_std.loc[
        country_std["country_standardisation_status"].ne("resolved")
    ].copy()
    province_dups = duplicate_rows(province_std, "province_match_key")
    country_dups = duplicate_rows(country_std, "country_match_key")

    save_csv(
        province_std,
        out_dir / "standardised_unique_province_centroid_queries.csv",
    )
    save_csv(
        country_std,
        out_dir / "standardised_unique_country_centroid_queries.csv",
    )
    save_csv(province_issues, out_dir / "province_standardisation_issues.csv")
    save_csv(country_issues, out_dir / "country_standardisation_issues.csv")
    save_csv(
        province_dups, out_dir / "duplicate_standardised_province_keys.csv"
    )
    save_csv(
        country_dups, out_dir / "duplicate_standardised_country_keys.csv"
    )

    summary: list[dict[str, object]] = []
    summary.extend([
        summary_row("province", "input_combinations", len(province_std)),
        summary_row(
            "province", "ready_for_matching",
            province_std["province_standardisation_status"].eq(
                "ready_for_natural_earth_matching"
            ).sum(),
        ),
        summary_row(
            "province", "ready_but_abbreviation_needs_attention",
            province_std["province_standardisation_status"].eq(
                "ready_but_abbreviation_needs_attention"
            ).sum(),
        ),
        summary_row(
            "province", "country_not_resolved",
            province_std["province_standardisation_status"].eq(
                "country_not_resolved"
            ).sum(),
        ),
        summary_row("province", "issue_rows", len(province_issues)),
        summary_row("province", "rows_in_duplicate_match_keys", len(province_dups)),
        summary_row("country", "input_combinations", len(country_std)),
        summary_row(
            "country", "ready_for_matching",
            country_std["country_ready_for_natural_earth_matching"].sum(),
        ),
        summary_row("country", "issue_rows", len(country_issues)),
        summary_row("country", "rows_in_duplicate_match_keys", len(country_dups)),
    ])

    if no_country_path.exists():
        no_country = read_csv(no_country_path)
        nc_province_col = find_column(
            no_country, PROVINCE_COL_ALIASES, required=True
        )
        no_country["province_state_original_checked"] = no_country[
            nc_province_col
        ].map(clean_cell)
        no_country["province_state_normalised_key"] = no_country[
            nc_province_col
        ].map(ascii_key)
        no_country["standardisation_status"] = (
            "excluded_missing_country_identifier"
        )
        save_csv(
            no_country,
            out_dir / "province_without_country_identifier_checked.csv",
        )
        summary.append(
            summary_row(
                "province_without_country", "excluded_rows", len(no_country)
            )
        )

    summary_df = pd.DataFrame(summary)
    save_csv(summary_df, out_dir / "centroid_query_standardisation_summary.csv")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nOutputs saved to:", out_dir.resolve())
    print("\nImportant:")
    print("- No centroid coordinates were calculated.")
    print("- Original columns were preserved.")
    print("- Review all issue and duplicate-key files before Natural Earth matching.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
