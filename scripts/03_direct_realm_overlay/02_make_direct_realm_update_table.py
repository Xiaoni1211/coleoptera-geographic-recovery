#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


INPUT_FILE = Path(
    "direct_coordinate_realm_assignment/"
    "direct_coordinate_realm_assignments.csv"
)

OUTPUT_FILE = Path(
    "direct_coordinate_realm_assignment/"
    "direct_coordinate_realm_update_table.csv"
)

KEEP_COLUMNS = [
    "input_row_number",
    "BTseq_id",
    "assigned_realm",
    "realm_assignment_method",
    "realm_confidence",
    "final_coordinate_source_standardised",
    "ptp_evidence_eligible",
]


def main():
    print("=" * 80)
    print("CREATE DIRECT REALM UPDATE TABLE")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=KEEP_COLUMNS + ["realm_overlay_status"],
        low_memory=False,
    )

    print(f"Rows in assignment file: {len(df):,}")

    # 只保留成功分配到realm的记录
    update = df.loc[
        df["realm_overlay_status"].eq("assigned")
        & df["assigned_realm"].notna(),
        KEEP_COLUMNS,
    ].copy()

    print(f"Successful realm updates: {len(update):,}")

    # 检查主数据行号是否重复
    duplicated_rows = update["input_row_number"].duplicated().sum()

    if duplicated_rows:
        raise ValueError(
            f"Found {duplicated_rows:,} duplicated input_row_number values. "
            "Update table was not saved."
        )

    # 检查BTseq_id缺失情况
    missing_ids = update["BTseq_id"].isna().sum()

    # 按主文件中的原始行号排序
    update = update.sort_values(
        "input_row_number"
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    update.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Update records:             {len(update):,}")
    print(f"Unique input row numbers:   {update['input_row_number'].nunique():,}")
    print(f"Unique BTseq_id values:     {update['BTseq_id'].nunique(dropna=True):,}")
    print(f"Missing BTseq_id values:    {missing_ids:,}")
    print()

    print("Realm assignments:")
    print(
        update["assigned_realm"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Confidence levels:")
    print(
        update["realm_confidence"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Assignment methods:")
    print(
        update["realm_assignment_method"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print(f"Saved update table: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
