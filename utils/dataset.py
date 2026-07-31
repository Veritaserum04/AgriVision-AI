import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# -------------------------------
# Device
# -------------------------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

# -------------------------------
# Image Transformations
# -------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Dataset
# -------------------------------

dataset = datasets.ImageFolder(
    root="dataset/PlantVillage",
    transform=train_transform
)

NUM_CLASSES = len(dataset.classes)

print(f"\nTotal Images : {len(dataset)}")
print(f"Total Classes: {NUM_CLASSES}")

print("\nClasses:\n")

for index, cls in enumerate(dataset.classes):
    print(f"{index:2d}. {cls}")

# -------------------------------
# Train / Validation Split
# -------------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

val_dataset.dataset.transform = val_transform

# -------------------------------
# Data Loaders
# -------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
print("\nDataset Loaded Successfully!")

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")