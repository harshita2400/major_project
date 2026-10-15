
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(r"E:\major\data\parquet\preprocessed")

splits = {}

for name in ["train_balanced", "validation", "test"]:
    path = DATA_DIR / f"{name}.parquet"

    df = pd.read_parquet(
        path,
        columns=["file_name", "caption"]
    )

    splits[name] = df

    print(f"\n{name}")
    print("Rows:", len(df))
    print("Classes:", df["caption"].nunique())
    print("Unique filenames:", df["file_name"].nunique())

    if name == "train_balanced":
        print(
            "Training class-count range:",
            df["caption"].value_counts().min(),
            "to",
            df["caption"].value_counts().max()
        )

# Check for filename overlap between splits.
for a, b in [
    ("train_balanced", "validation"),
    ("train_balanced", "test"),
    ("validation", "test"),
]:
    files_a = set(splits[a]["file_name"].dropna())
    files_b = set(splits[b]["file_name"].dropna())

    overlap = files_a & files_b

    print(f"\nOverlap: {a} vs {b} = {len(overlap)}")

# Inspect only a small batch of image records.
path = DATA_DIR / "train_balanced.parquet"
parquet = pq.ParquetFile(path)

batch = next(
    parquet.iter_batches(
        batch_size=10,
        columns=["image"]
    )
)

images = batch.column(0)

print("\nImage column type:", images.type)

bytes_field = images.field("bytes")
path_field = images.field("path")

print("Bytes missing in sample:", bytes_field.null_count)
print("Paths missing in sample:", path_field.null_count)

for i in range(min(3, len(images))):
    image_bytes = bytes_field[i].as_py()
    image_path = path_field[i].as_py()

    print(f"\nImage {i}")
    print(
        "Embedded byte length:",
        len(image_bytes) if image_bytes is not None else None
    )
    print("Stored path:", image_path)
    print(
        "Path exists:",
        Path(image_path).is_file()
        if image_path else False
    )
