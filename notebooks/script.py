
from pathlib import Path
import pandas as pd

# Folder containing your Parquet files
DATA_DIR = Path(r"E:\major\data\parquet")

# Find all Parquet files in this folder
files = sorted(DATA_DIR.glob("*.parquet"))

print("Parquet files found:", len(files))

for file in files:
    print(file.name)

if not files:
    raise FileNotFoundError(
        f"No .parquet files found in {DATA_DIR}"
    )

# Load only metadata first; don't load image bytes yet
df = pd.concat(
    [
        pd.read_parquet(
            file,
            columns=["file_name", "caption"]
        )
        for file in files
    ],
    ignore_index=True
)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

print("\nNumber of unique captions:", df["caption"].nunique())

print("\nMost common classes/captions:")
print(df["caption"].value_counts().head(20))

print("\nLeast common classes/captions:")
print(df["caption"].value_counts().tail(20))
