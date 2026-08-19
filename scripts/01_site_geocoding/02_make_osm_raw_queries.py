import pandas as pd

df = pd.read_csv("unique_precise_site_queries.csv")


def clean(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in ["nan", "na", "none", "null"]:
        return ""
    return x


def build_query(row):
    parts = [
        clean(row["site"]),
        clean(row["sector"]),
        clean(row["region"]),
        clean(row["province_state"]),
        clean(row["country"])
    ]

    parts = [p for p in parts if p]

    return ", ".join(parts)


df["raw_query"] = df.apply(build_query, axis=1)

print(df[["site", "raw_query"]].head(20))

df.to_csv("unique_precise_site_queries_raw.csv", index=False)

print(f"\nSaved {len(df)} queries.")
