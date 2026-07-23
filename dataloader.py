import os
import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.2, 1.0),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def get_dataloaders(
        data_dir: str, batch_size: int, n_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    trainset = ImageFolder(data_dir, transform=train_transform)
    valset = ImageFolder(data_dir, transform=test_transform)
    testset = ImageFolder(data_dir, transform=test_transform)

    if os.path.exists(f"{data_dir}/splits.pt"):
        splits = torch.load(f"{data_dir}/splits.pt")
        train_set = Subset(trainset, splits["train"])
        val_set = Subset(valset, splits["val"])
        test_set = Subset(testset, splits["test"])
    else:
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(trainset.targets):
            class_to_indices[label].append(idx)
        for cls, indices in class_to_indices.items():
            if len(indices) < 10:
                raise ValueError(f"Class {cls} has only {len(indices)} images.")
        val_indices = []
        remaining_indices = []
        for indices in class_to_indices.values():
            random.shuffle(indices)
            val_indices.extend(indices[:10])
            remaining_indices.extend(indices[10:])
        random.shuffle(remaining_indices)
        train_indices = remaining_indices[:500_000]
        test_indices = remaining_indices[500_000:]
        assert len(train_indices) == 500_000,\
            f"Expected 500,000 training images, but got {len(train_indices)}"
        assert len(val_indices) == 10_000,\
            f"Expected 10,000 validation images, but got {len(val_indices)}"
        assert len(test_indices) > 29_800,\
            f"Expected approx 30,000 test images, but got {len(test_indices)}"
        train_set = Subset(trainset, train_indices)
        val_set = Subset(valset, val_indices)
        test_set = Subset(testset, test_indices)

        torch.save({
            "train": train_indices, "val": val_indices, "test": test_indices,
        }, f"{data_dir}/splits.pt")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader
