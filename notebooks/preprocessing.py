
from pathlib import Path
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(r"E:\major\data\parquet")
OUT_DIR = DATA_DIR / "preprocessed"

MIN_SAMPLES = 500
MAX_SAMPLES = 1000
SEED = 42
BATCH_SIZE = 128

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted(DATA_DIR.glob("*.parquet"))

    if len(files) != 7:
        raise ValueError(
            f"Expected 7 Parquet files in {DATA_DIR}, found {len(files)}"
        )

    # --------------------------------------------------------
    # PASS 1: Read metadata only (no image bytes).
    # Store source file and original row position.
    # --------------------------------------------------------
    print("PASS 1: Reading metadata...")

    metadata_parts = []

    for file_index, file in enumerate(files):
        print(f"Reading metadata: {file.name}")

        metadata = pd.read_parquet(
            file,
            columns=["file_name", "caption"]
        )

        metadata["_file_index"] = file_index
        metadata["_row_index"] = range(len(metadata))
        metadata_parts.append(metadata)

    meta = pd.concat(metadata_parts, ignore_index=True)

    original_rows = len(meta)
    meta = meta.dropna(subset=["file_name", "caption"]).copy()

    print(f"Original rows: {original_rows:,}")
    print(f"Rows with valid metadata: {len(meta):,}")

    # Save original class counts.
    original_counts = meta["caption"].value_counts()

    original_counts.rename_axis("caption").reset_index(
        name="count"
    ).to_csv(OUT_DIR / "class_counts_before.csv", index=False)

    # --------------------------------------------------------
    # FILTER RARE CLASSES
    # --------------------------------------------------------
    retained_labels = original_counts[
        original_counts >= MIN_SAMPLES
    ].index

    meta = meta[meta["caption"].isin(retained_labels)].copy()

    # Remove repeated filenames.
    meta = meta.drop_duplicates(
        subset=["file_name"],
        keep="first"
    ).copy()

    print(f"Classes after rare-class filtering: {meta['caption'].nunique()}")
    print(f"Rows after rare-class filtering: {len(meta):,}")

    if meta["caption"].nunique() < 2:
        raise ValueError("Fewer than two classes remain.")

    # --------------------------------------------------------
    # SPLIT FIRST: 70% train, 15% validation, 15% test.
    # --------------------------------------------------------
    print("\nSplitting into train, validation and test...")

    train_val, test = train_test_split(
        meta,
        test_size=0.15,
        random_state=SEED,
        stratify=meta["caption"]
    )

    train, validation = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=SEED,
        stratify=train_val["caption"]
    )

    # --------------------------------------------------------
    # BALANCE TRAINING DATA ONLY
    # Randomly select at most 1,000 rows per class.
    # --------------------------------------------------------
    print("Balancing training classes...")

    balanced_parts = []

    for i, (_, group) in enumerate(
        train.groupby("caption", sort=True)
    ):
        sample_size = min(MAX_SAMPLES, len(group))

        balanced_parts.append(
            group.sample(
                n=sample_size,
                replace=False,
                random_state=SEED + i
            )
        )

    train_balanced = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    train_balanced = train_balanced.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    splits = {
        "train_balanced": train_balanced,
        "validation": validation,
        "test": test
    }

    # Save the planned class counts before extracting image bytes.
    report = pd.concat(
        [
            train_balanced["caption"].value_counts().rename("train_count"),
            validation["caption"].value_counts().rename("validation_count"),
            test["caption"].value_counts().rename("test_count")
        ],
        axis=1
    ).fillna(0).astype(int)

    report.index.name = "caption"
    report.to_csv(OUT_DIR / "class_counts_after.csv")

    # --------------------------------------------------------
    # Build lookup:
    # (source file index, original row index) -> split name
    # --------------------------------------------------------
    selected = {}

    for split_name, frame in splits.items():
        selected[split_name] = {}

        # Use positional tuples to avoid underscore-column renaming.
        for file_index, row_index in frame[
            ["_file_index", "_row_index"]
        ].itertuples(index=False, name=None):
            key = (int(file_index), int(row_index))
            selected[split_name][key] = True

    # --------------------------------------------------------
    # PASS 2: Stream image data in batches.
    # Use PyArrow ParquetWriter to avoid holding all images in RAM.
    # --------------------------------------------------------
    print("\nPASS 2: Extracting selected images in batches...")

    output_paths = {
        name: OUT_DIR / f"{name}.parquet"
        for name in splits
    }

    writers = {name: None for name in splits}
    rows_written = {name: 0 for name in splits}

    try:
        for file_index, file in enumerate(files):
            print(f"\nProcessing image batches: {file.name}")

            parquet_file = pq.ParquetFile(file)
            row_offset = 0

            for batch in parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=["file_name", "image", "caption"]
            ):
                batch_rows = batch.num_rows

                # Find selected rows in this batch for each split.
                for split_name in splits:
                    local_indices = []

                    for local_index in range(batch_rows):
                        key = (file_index, row_offset + local_index)

                        if key in selected[split_name]:
                            local_indices.append(local_index)

                    if not local_indices:
                        continue

                    # Select only the required image rows.
                    index_array = pa.array(
                        local_indices,
                        type=pa.int64()
                    )
                    selected_table = batch.take(index_array)

                    if writers[split_name] is None:
                        writers[split_name] = pq.ParquetWriter(
                            output_paths[split_name],
                            selected_table.schema,
                            compression="snappy"
                        )

                    writers[split_name].write_table(
                        pa.Table.from_batches([selected_table])
)
                    rows_written[split_name] += selected_table.num_rows

                row_offset += batch_rows

            print(f"Finished {file.name}")

    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    # --------------------------------------------------------
    # SUMMARY AND VALIDATION
    # --------------------------------------------------------
    summary = {
        "input_files": [file.name for file in files],
        "original_rows": int(original_rows),
        "rows_after_metadata_filtering": int(len(meta)),
        "classes_retained": int(meta["caption"].nunique()),
        "minimum_class_samples": MIN_SAMPLES,
        "maximum_training_samples_per_class": MAX_SAMPLES,
        "random_seed": SEED,
        "batch_size": BATCH_SIZE,
        "rows_written": rows_written,
        "output_directory": str(OUT_DIR)
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n========== PREPROCESSING COMPLETE ==========")
    print(json.dumps(summary, indent=2))

    # Confirm output row counts.
    print("\nVerifying saved Parquet files:")

    for name, path in output_paths.items():
        if not path.exists():
            raise RuntimeError(f"Output file was not created: {path}")

        saved_rows = pq.ParquetFile(path).metadata.num_rows

        print(f"{path.name}: {saved_rows:,} rows")

        if saved_rows != rows_written[name]:
            raise RuntimeError(
                f"Row count mismatch for {name}: "
                f"expected {rows_written[name]}, got {saved_rows}"
            )

    print("\nAll output row counts verified.")


if __name__ == "__main__":
    main()
