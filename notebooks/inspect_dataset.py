
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(r"E:\major\data\parquet\preprocessed")

for name in ["train_balanced", "validation", "test"]:
    path = DATA_DIR / f"{name}.parquet"

    print(f"\n{'=' * 55}")
    print(f"Dataset: {name}")

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    parquet = pq.ParquetFile(path)
    print("Rows:", parquet.metadata.num_rows)
    print("Schema:")
    print(parquet.schema)

    # Read metadata first; avoid loading all image bytes.
    df = pd.read_parquet(
        path,
        columns=["file_name", "caption"]
    )

    print("Unique classes:", df["caption"].nunique())
    print("Missing filenames:", df["file_name"].isna().sum())
    print("Missing captions:", df["caption"].isna().sum())

    print("\nMost common classes:")
    print(df["caption"].value_counts().head(5))

    print("\nExample captions:")
    for caption in df["caption"].drop_duplicates().head(5):
        print("-", caption)

    print("\nExample filenames:")
    print(df["file_name"].head(3).to_list())
