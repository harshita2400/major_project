
from pathlib import Path
from io import BytesIO

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(r"E:\major\data\parquet\preprocessed")
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 2
MAX_TRAIN_SAMPLES = 5000
NUM_WORKERS = 0

print("Device:", DEVICE)


def load_records(path, limit=None):
    parquet = pq.ParquetFile(path)
    records = []

    for batch in parquet.iter_batches(
        batch_size=256,
        columns=["image", "caption"]
    ):
        images = batch.column(0)
        captions = batch.column(1)

        for i in range(batch.num_rows):
            image_bytes = images.field("bytes")[i].as_py()
            caption = captions[i].as_py()

            if image_bytes is None or caption is None:
                continue

            records.append((image_bytes, caption))

            if limit is not None and len(records) >= limit:
                return records

    return records


class LeafDataset(Dataset):
    def __init__(self, records, label_encoder, training=False):
        self.records = records
        self.label_encoder = label_encoder

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip()
            if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_bytes, caption = self.records[index]

        with Image.open(BytesIO(image_bytes)) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        label = int(self.label_encoder.transform([caption])[0])

        return image, label


def main():
    train_path = DATA_DIR / "train_balanced.parquet"
    val_path = DATA_DIR / "validation.parquet"

    # Load a small training subset for the initial smoke test.
    train_records = load_records(
        train_path,
        limit=MAX_TRAIN_SAMPLES
    )
    val_records = load_records(val_path)

    print("Training samples:", len(train_records))
    print("Validation samples:", len(val_records))

    train_labels = sorted({
        caption for _, caption in train_records
    })

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)

    known_labels = set(label_encoder.classes_)
    val_records = [
        (img, caption)
        for img, caption in val_records
        if caption in known_labels
    ]

    train_ds = LeafDataset(
        train_records, label_encoder, training=True
    )
    val_ds = LeafDataset(
        val_records, label_encoder, training=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(label_encoder.classes_)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Validation accuracy: {accuracy:.4f}"
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": label_encoder.classes_.tolist()
    }, DATA_DIR / "centralized_baseline.pt")

    print("Saved centralized_baseline.pt")


if __name__ == "__main__":
    main()
