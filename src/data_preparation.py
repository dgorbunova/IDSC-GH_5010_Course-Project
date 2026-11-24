
import pandas as pd
from pathlib import Path

# file paths
project_dir = Path(__file__).resolve().parent.parent
input_path = project_dir / "data" / "hotel_bookings.csv"
output_path = project_dir / "data" / "hotel_bookings_cleaned.csv"

# original dataset
df = pd.read_csv(input_path)
df_clean = df.copy()

# missing values + categorical for company and agent
df_clean["children"] = df_clean["children"].fillna(0).astype(int)
df_clean["country"] = df_clean["country"].fillna("Unknown")

df_clean["has_agent"] = df_clean["agent"].notna()
df_clean["has_company"] = df_clean["company"].notna()

df_clean = df_clean.drop(columns=["agent", "company"])

# retain exact duplicate records


# remove invalid records
zero_guests = (
    df_clean["adults"] +
    df_clean["children"] +
    df_clean["babies"] == 0
)

zero_nights = (
    df_clean["stays_in_week_nights"] +
    df_clean["stays_in_weekend_nights"] == 0
)

negative_adr = df_clean["adr"] < 0

invalid_mask = zero_guests | zero_nights | negative_adr
df_clean = df_clean.loc[~invalid_mask].copy()

# remove the extreme ADR outlier
df_clean = df_clean.loc[df_clean["adr"] <= 1000].copy()

# standardise meal categories
df_clean["meal"] = df_clean["meal"].replace("Undefined", "SC")

# remov known outcome-leakage columns
df_clean = df_clean.drop(
    columns=["reservation_status", "reservation_status_date"]
)

# checks
assert df_clean.shape == (118563, 30)
assert df_clean.isna().sum().sum() == 0
assert "is_canceled" in df_clean.columns

df_clean.to_csv(output_path, index=False)

print("Original dataset:", df.shape)
print("Cleaned dataset:", df_clean.shape)
print("Rows removed:", len(df) - len(df_clean))
print("Remaining missing values:", df_clean.isna().sum().sum())
print(f"Saved to: {output_path}")
