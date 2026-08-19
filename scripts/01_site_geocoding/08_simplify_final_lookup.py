import pandas as pd

INPUT = "final_site_geocoding_lookup.csv"
OUTPUT = "final_site_geocoding_lookup_simple.csv"

print("Loading final lookup...")

df = pd.read_csv(INPUT, dtype=str, low_memory=False)

# ============================================================
# Columns to keep
# ============================================================

keep_cols = [
    "country",
    "province_state",
    "region",
    "sector",
    "site",
    "raw_query",
    "final_success",
    "final_geocode_source",
    "final_query_used",
    "final_lat",
    "final_lon",
    "final_address_or_display_name",
]

missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input file: {missing}")

df_simple = df[keep_cols].copy()

# ============================================================
# Add geocoding level
# ============================================================

# 当前这个 lookup 只包含 site-level geocoding：
# OSM raw / OSM clean / Google fallback 都是基于 site query 的结果。
df_simple["final_geocoding_level"] = "site"

# 如果有失败记录，保留为 failed，避免误认为成功 site-level 坐标
df_simple.loc[
    df_simple["final_success"].astype(str).str.lower() != "true",
    "final_geocoding_level"
] = "failed"

# ============================================================
# Reorder columns
# ============================================================

final_cols = [
    "country",
    "province_state",
    "region",
    "sector",
    "site",
    "raw_query",
    "final_success",
    "final_geocoding_level",
    "final_geocode_source",
    "final_query_used",
    "final_lat",
    "final_lon",
    "final_address_or_display_name",
]

df_simple = df_simple[final_cols]

# ============================================================
# Save
# ============================================================

df_simple.to_csv(OUTPUT, index=False)

# ============================================================
# Summary
# ============================================================

print("\n===== Simplified lookup summary =====")
print(f"Rows: {len(df_simple):,}")
print(f"Columns: {len(df_simple.columns)}")

print("\nfinal_success:")
print(df_simple["final_success"].value_counts(dropna=False))

print("\nfinal_geocoding_level:")
print(df_simple["final_geocoding_level"].value_counts(dropna=False))

print("\nfinal_geocode_source:")
print(df_simple["final_geocode_source"].value_counts(dropna=False))

print(f"\nSaved to: {OUTPUT}")
print("Done.")
