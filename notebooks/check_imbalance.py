
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(r"E:\major\data\parquet")
files = sorted(DATA_DIR.glob("*.parquet"))

# Read only metadata, not the image bytes
df = pd.concat(
    [
        pd.read_parquet(f, columns=["file_name", "caption"])
        for f in files
    ],
    ignore_index=True
)

counts = df["caption"].value_counts()

print("Total images:", len(df))
print("Total classes:", len(counts))

print("\nClasses remaining at different thresholds:")
for threshold in [100, 250, 500, 1000, 1500]:
    remaining = counts[counts >= threshold]
    print(
        f"Minimum {threshold:4} images: "
        f"{len(remaining):2} classes, "
        f"{remaining.sum():,} images retained"
    )

# Save complete class-count report
report = counts.rename_axis("caption").reset_index(name="count")
report.to_csv(DATA_DIR / "class_counts.csv", index=False)

# Plot class imbalance
plt.figure(figsize=(14, 5))
counts.sort_values(ascending=False).plot(kind="bar")
plt.title("Class Distribution Before Preprocessing")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(DATA_DIR / "class_imbalance.png", dpi=150)
plt.show()

print("\nSaved class_counts.csv and class_imbalance.png")
