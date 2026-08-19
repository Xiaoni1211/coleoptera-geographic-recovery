import pandas as pd

input_file = "matched_o_db_id.csv"
output_file = "unique_locations.csv"

df = pd.read_csv(input_file)

# 找出缺失 latitude 或 longitude 的记录
missing = df["latitude"].isna() | df["longitude"].isna()

# 用这些字段代表一个“地点”
cols = [
    "country",
    "province_state_territory",
    "district_country_shire",
    "locality",
    "precise_location"
]

# 只保留文件中真实存在的列
cols = [c for c in cols if c in df.columns]

unique_locations = (
    df.loc[missing, cols]
    .fillna("")
    .drop_duplicates()
    .reset_index(drop=True)
)

# 加一个 location_id，方便之后匹配回原表
unique_locations.insert(0, "location_id", range(1, len(unique_locations) + 1))

# 生成 geocode 查询字符串
def make_query(row):
    parts = []
    for c in cols:
        value = str(row[c]).strip()
        if value and value.upper() != "NA":
            parts.append(value)
    return ", ".join(parts)

unique_locations["geocode_query"] = unique_locations.apply(make_query, axis=1)

unique_locations.to_csv(output_file, index=False)

print("Missing coordinate rows:", missing.sum())
print("Unique missing locations:", len(unique_locations))
print("Saved to:", output_file)
