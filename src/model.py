# src/train_vit_timm.py

import os
import argparse

import torch
from torch import nn
from torch.optim import AdamW

import timm  # pretrained vision models

from dataset import get_dataloaders


def compute_accuracy(logits, targets):
    """
    logits: (B, num_classes)
    targets: (B,)
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(logits, labels)
        running_loss += loss.item()
        running_acc += acc
        n_batches += 1

    return running_loss / n_batches, running_acc / n_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            acc = compute_accuracy(logits, labels)

            running_loss += loss.item()
            running_acc += acc
            n_batches += 1

    return running_loss / n_batches, running_acc / n_batches


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # load data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)
    print(f"Classes: {class_names} (num_classes={num_classes})")

    # create ViT model from timm
    print(f"Creating model: {args.model_name} (pretrained={args.pretrained})")
    model = timm.create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=num_classes,
    )
    model.to(device)

    # optionally freeze backbone and only train classifier head
    if args.freeze_backbone:
        print("Freezing backbone parameters, training only classifier head.")
        for name, param in model.named_parameters():
            if "head" not in name and "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.44f}"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best_vit_timm.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ➜ Saved new best model to {ckpt_path}")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze all but head")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if GPU is available")

    args = parser.parse_args()
    main(args)
