import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import BCCDDataset
from utils import get_device, load_detection_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on the BCCD dataset")
    parser.add_argument("--dataset_dir", type=str, default="BCCD_Dataset", help="Path to the BCCD dataset root")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("checkpoints", "bccd_fasterrcnn.pth"),
        help="Where to save the trained checkpoint",
    )
    parser.add_argument("--resize", type=int, default=512, help="Resize images to a square size before training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split")
    return parser.parse_args()


def collate_fn(batch):
    return tuple(zip(*batch))


def split_dataset(dataset, seed: int):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    split_index = int(0.8 * len(indices))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_loaders(dataset_dir: str, resize: int, batch_size: int, seed: int):
    dataset = BCCDDataset(root_dir=dataset_dir, resize_to=(resize, resize))
    train_dataset, val_dataset = split_dataset(dataset, seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def freeze_backbone_for_fast_training(model):
    """Freeze most of the network so CPU training is lighter for assignment use."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def train_one_epoch(model, optimizer, dataloader, device, epoch_number: int):
    model.train()
    running_loss = 0.0

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = [image.to(device) for image in images]
        batch_targets = []
        for target in targets:
            batch_targets.append(
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                    "image_id": target["image_id"].to(device),
                    "area": target["area"].to(device),
                    "iscrowd": target["iscrowd"].to(device),
                }
            )

        loss_dict = model(images, batch_targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += float(total_loss.item())

        if batch_index % 20 == 0 or batch_index == len(dataloader):
            print(
                f"Epoch {epoch_number} | Batch {batch_index}/{len(dataloader)} | "
                f"Loss: {float(total_loss.item()):.4f}"
            )

    return running_loss / max(1, len(dataloader))


def validate_one_epoch(model, dataloader, device):
    model.train()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            batch_targets = []
            for target in targets:
                batch_targets.append(
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                        "image_id": target["image_id"].to(device),
                        "area": target["area"].to(device),
                        "iscrowd": target["iscrowd"].to(device),
                    }
                )

            loss_dict = model(images, batch_targets)
            total_loss = sum(loss for loss in loss_dict.values())
            running_loss += float(total_loss.item())

    return running_loss / max(1, len(dataloader))


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = create_loaders(
        dataset_dir=args.dataset_dir,
        resize=args.resize,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = load_detection_model(checkpoint_path=None, device=device, verbose=False)
    freeze_backbone_for_fast_training(model)

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    best_val_loss = float("inf")

    for epoch_number in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch_number)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch_number} complete | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Saved checkpoint to: {output_path}")

    print("Training finished.")
    print(f"Best checkpoint: {output_path}")


if __name__ == "__main__":
    main()