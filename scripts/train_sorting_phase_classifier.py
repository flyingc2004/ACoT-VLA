#!/usr/bin/env python3
"""Train a CNN/ViT classifier for sorting continuous phase recognition.

Expected classes (folder names under train/ and val/):
- already_reset
- in_progress
- task_terminal
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


EXPECTED_LABELS = ("already_reset", "in_progress", "task_terminal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sorting phase classifier")
    parser.add_argument(
        "--data-dir",
        default="/home/xhz/ljz/ACoT-VLA/outputs/sorting_phase_dataset",
        help="Dataset root with train/ and val/",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/xhz/ljz/ACoT-VLA/outputs/sorting_phase_classifier",
        help="Directory to save checkpoints and metrics",
    )
    parser.add_argument("--arch", choices=["resnet18", "vit_b_16"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device selection. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated CUDA device ids for DataParallel, e.g. 0,1,2,3. Empty means all visible GPUs.",
    )
    parser.add_argument(
        "--disable-multi-gpu",
        action="store_true",
        help="Disable DataParallel even when multiple GPUs are visible.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Print one text log every N train steps.",
    )
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument(
        "--enforce-target-accuracy",
        action="store_true",
        help="Exit with non-zero code if best val accuracy < target",
    )
    parser.add_argument(
        "--save-name",
        default="sorting_phase_classifier_best.pt",
        help="Checkpoint filename",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "vit_b_16":
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported arch: {arch}")


def resolve_device_and_parallel(model: nn.Module, args: argparse.Namespace) -> tuple[nn.Module, torch.device]:
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    if args.device == "cpu" or (args.device == "auto" and not cuda_available):
        print("[INFO] Running on CPU")
        return model.to(torch.device("cpu")), torch.device("cpu")

    if args.device == "cuda" and not cuda_available:
        raise RuntimeError("--device=cuda but CUDA is not available")

    if args.gpu_ids.strip():
        device_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        device_ids = list(range(cuda_count))

    if not device_ids:
        raise RuntimeError("No CUDA device ids resolved for training")

    device = torch.device(f"cuda:{device_ids[0]}")
    model = model.to(device)

    using_dp = (not args.disable_multi_gpu) and len(device_ids) > 1
    if using_dp:
        model = nn.DataParallel(model, device_ids=device_ids)

    visible_names = [torch.cuda.get_device_name(i) for i in range(cuda_count)]
    print(f"[INFO] CUDA available: {cuda_available}, total visible GPUs: {cuda_count}")
    print(f"[INFO] Visible GPU names: {visible_names}")
    print(f"[INFO] Selected device ids: {device_ids}")
    print(f"[INFO] DataParallel enabled: {using_dp}")

    return model, device


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += float(loss.item()) * targets.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_count += int(targets.size(0))

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/ and val/ under {data_root}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = tv_transforms.Compose(
        [
            tv_transforms.Resize((args.image_size, args.image_size)),
            tv_transforms.RandomHorizontalFlip(p=0.5),
            tv_transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = tv_transforms.Compose(
        [
            tv_transforms.Resize((args.image_size, args.image_size)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    class_names = train_dataset.classes
    if tuple(class_names) != tuple(sorted(EXPECTED_LABELS)):
        print(
            "[WARN] Dataset class names differ from expected labels. "
            f"found={class_names}, expected={EXPECTED_LABELS}"
        )

    if class_names != val_dataset.classes:
        raise ValueError(f"Train/val class mapping mismatch: {class_names} vs {val_dataset.classes}")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Class names: {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = build_model(args.arch, num_classes=len(class_names), pretrained=args.pretrained)
    model, device = resolve_device_and_parallel(model, args)

    class_counts = np.bincount(np.array(train_dataset.targets), minlength=len(class_names))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for step_idx, (images, targets) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            batch_size = targets.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_correct += int((preds == targets).sum().item())
            train_count += int(batch_size)

            pbar.set_postfix(
                loss=f"{train_loss_sum / max(train_count, 1):.4f}",
                acc=f"{train_correct / max(train_count, 1):.4f}",
            )

            if step_idx % max(args.log_interval, 1) == 0 or step_idx == len(train_loader):
                print(
                    f"[TRAIN] epoch={epoch}/{args.epochs} step={step_idx}/{len(train_loader)} "
                    f"loss={train_loss_sum / max(train_count, 1):.4f} "
                    f"acc={train_correct / max(train_count, 1):.4f}",
                    flush=True,
                )

        scheduler.step()

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=True))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {
                "arch": args.arch,
                "class_names": class_names,
                "image_size": args.image_size,
                "mean": mean,
                "std": std,
                "val_acc": best_val_acc,
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "args": vars(args),
            }
            ckpt_path = out_root / args.save_name
            torch.save(best_state, ckpt_path)
            print(f"[INFO] Saved best checkpoint to {ckpt_path} (val_acc={best_val_acc:.4f})")

    metrics_path = out_root / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val_acc, "history": history}, f, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Target accuracy: {args.target_accuracy:.4f}")
    if best_val_acc >= args.target_accuracy:
        print("[INFO] Target accuracy reached.")
    else:
        print("[WARN] Target accuracy not reached.")
        if args.enforce_target_accuracy:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
