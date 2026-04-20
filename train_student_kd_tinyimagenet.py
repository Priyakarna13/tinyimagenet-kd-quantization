import os
import json
import time
import random
import csv
import argparse
from dataclasses import asdict, dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, Dataset
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

SAVE_DIR = "/data/pjiang18/SLP/ModelCompression"
DATA_DIR = f"{SAVE_DIR}/data/tiny-imagenet-200"
CHECKPOINT_DIR = f"{SAVE_DIR}/checkpoints"
RESULTS_DIR = f"{SAVE_DIR}/results"
LOG_DIR = f"{SAVE_DIR}/logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_TEACHER_CKPT = f"{CHECKPOINT_DIR}/tinyimagenet_resnet50_teacher_best.pth"
DEFAULT_TEACHER_RESULTS = f"{RESULTS_DIR}/tinyimagenet_teacher_results.json"
DEFAULT_STUDENT_CKPT = f"{CHECKPOINT_DIR}/tinyimagenet_resnet18_student_best.pth"
DEFAULT_RESULTS_FILE = f"{RESULTS_DIR}/tinyimagenet_kd_student_results.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(labels: List[int], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def save_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_state_dict_flexible(model: nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
    model.load_state_dict(ckpt)
    return {"model_state_dict": ckpt}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast():
            outputs = model(inputs)

        loss = criterion(outputs.float(), targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def build_teacher(num_classes: int = 200) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_student(num_classes: int = 200) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_lr_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    def lr_lambda(epoch_idx: int):
        if epochs <= 1:
            return 1.0
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        if epochs <= warmup_epochs:
            return 1.0
        denom = max(1, epochs - warmup_epochs - 1)
        progress = (epoch_idx - warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TinyImageNetBenchmarkDataset(Dataset):
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.root = root
        self.transform = transform
        self.class_to_idx = class_to_idx

        ann_path = os.path.join(root, "val_annotations.txt")
        images_dir = os.path.join(root, "images")

        if os.path.isfile(ann_path) and os.path.isdir(images_dir):
            self.samples = []
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    img_name, wnid = parts[0], parts[1]
                    if wnid not in self.class_to_idx:
                        continue
                    img_path = os.path.join(images_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[wnid]))
            self.targets = [y for _, y in self.samples]
            self.fallback = None
        else:
            self.fallback = torchvision.datasets.ImageFolder(root, transform=transform)
            self.samples = None
            self.targets = self.fallback.targets

    def __len__(self):
        if self.fallback is not None:
            return len(self.fallback)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.fallback is not None:
            return self.fallback[idx]

        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


@dataclass
class TrainConfig:
    epochs: int = 100
    patience: int = 15
    min_delta: float = 0.001
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    val_split: float = 0.1
    download: bool = False
    input_size: int = 224
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    temperature: float = 4.0
    alpha: float = 0.7
    feature_beta: float = 2.0
    teacher_ckpt: str = DEFAULT_TEACHER_CKPT
    teacher_results: str = DEFAULT_TEACHER_RESULTS
    student_ckpt: str = DEFAULT_STUDENT_CKPT
    results_file: str = DEFAULT_RESULTS_FILE


def forward_resnet_with_features(backbone: nn.Module, x: torch.Tensor):
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    feat = backbone.layer4(x)
    pooled = backbone.avgpool(feat)
    logits = backbone.fc(torch.flatten(pooled, 1))
    return logits, feat


def kd_loss(student_logits, teacher_logits, targets, temperature, alpha, hard_criterion):
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()

    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    hard_loss = hard_criterion(student_logits, targets)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss, soft_loss, hard_loss


def feature_distill_loss(student_feat_proj: torch.Tensor, teacher_feat: torch.Tensor):
    student_vec = F.adaptive_avg_pool2d(student_feat_proj, 1).flatten(1)
    teacher_vec = F.adaptive_avg_pool2d(teacher_feat.detach(), 1).flatten(1)

    student_vec = F.normalize(student_vec, dim=1)
    teacher_vec = F.normalize(teacher_vec, dim=1)

    return F.mse_loss(student_vec, teacher_vec)


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    best_val_loss: float, best_val_acc: float, best_epoch: int, cfg: TrainConfig):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "cfg": asdict(cfg),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--feature_beta", type=float, default=2.0)
    parser.add_argument("--teacher_ckpt", type=str, default=DEFAULT_TEACHER_CKPT)
    parser.add_argument("--teacher_results", type=str, default=DEFAULT_TEACHER_RESULTS)
    parser.add_argument("--student_ckpt", type=str, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
        download=args.download,
        input_size=224,
        warmup_epochs=5,
        label_smoothing=0.1,
        temperature=args.temperature,
        alpha=args.alpha,
        feature_beta=args.feature_beta,
        teacher_ckpt=args.teacher_ckpt,
        teacher_results=args.teacher_results,
        student_ckpt=args.student_ckpt,
        results_file=args.results_file,
    )

    set_seed(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"Training on: {torch.cuda.get_device_name(0)}")
    print("GPUs used: 1")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(cfg.input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_root = os.path.join(DATA_DIR, "train")
    benchmark_root = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not os.path.isdir(benchmark_root):
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_root}")

    base_train = torchvision.datasets.ImageFolder(train_root, transform=None)
    train_labels = [label for _, label in base_train.samples]
    train_idx, internal_val_idx = stratified_split(train_labels, val_ratio=cfg.val_split, seed=cfg.seed)

    train_dataset = torchvision.datasets.ImageFolder(train_root, transform=train_transform)
    internal_val_dataset = torchvision.datasets.ImageFolder(train_root, transform=eval_transform)
    benchmark_dataset = TinyImageNetBenchmarkDataset(
        benchmark_root,
        class_to_idx=train_dataset.class_to_idx,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = DataLoader(
        Subset(internal_val_dataset, internal_val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    benchmark_loader = DataLoader(
        benchmark_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    num_classes = len(train_dataset.classes)
    print(f"Train: {len(train_idx):,} | Val: {len(internal_val_idx):,} | Benchmark: {len(benchmark_dataset):,}")
    print(f"Classes: {num_classes}")

    if not os.path.exists(cfg.teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {cfg.teacher_ckpt}")

    teacher_backbone = build_teacher(num_classes=num_classes)
    load_state_dict_flexible(teacher_backbone, cfg.teacher_ckpt, device)
    teacher_backbone = teacher_backbone.to(device)
    teacher_backbone.eval()
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    print("Teacher loaded and frozen.")

    teacher_params = sum(p.numel() for p in teacher_backbone.parameters()) / 1e6

    student_backbone = build_student(num_classes=num_classes).to(device)
    student_projector = nn.Sequential(
        nn.Conv2d(512, 2048, kernel_size=1, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
    ).to(device)

    student_params = sum(p.numel() for p in student_backbone.parameters()) / 1e6
    print(f"Teacher: {teacher_params:.1f}M params")
    print(f"Student: {student_params:.1f}M params")
    print(f"Compression: {teacher_params / student_params:.2f}x fewer params")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = GradScaler()

    # Single learning rate everywhere for a fair, clean benchmark.
    optimizer = optim.SGD(
        list(student_backbone.parameters()) + list(student_projector.parameters()),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    scheduler = get_lr_scheduler(optimizer, epochs=cfg.epochs, warmup_epochs=cfg.warmup_epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    history = []
    total_train_time = 0.0
    peak_mem_bytes = 0

    teacher_benchmark_acc = None
    if os.path.exists(cfg.teacher_results):
        try:
            with open(cfg.teacher_results, "r") as f:
                teacher_results = json.load(f)
            teacher_benchmark_acc = teacher_results.get("benchmark_acc")
        except Exception:
            teacher_benchmark_acc = None

    checkpoint_path = cfg.student_ckpt
    results_json = cfg.results_file
    results_csv = os.path.join(RESULTS_DIR, "tinyimagenet_kd_student_history.csv")

    print(
        f"\nDistilling ResNet50 -> ResNet18 | T={cfg.temperature} | alpha={cfg.alpha} | beta={cfg.feature_beta}"
    )
    print(
        f"\n{'Epoch':>6} {'LR':>8} {'TrainLoss':>10} {'KD':>8} {'Feat':>8} "
        f"{'ValLoss':>9} {'ValAcc':>8} {'Img/s':>8} {'PeakGB':>8} {'Time':>6}"
    )
    print("-" * 104)

    global_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        student_backbone.train()
        student_projector.train()
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        running_total = 0.0
        running_kd = 0.0
        running_hard = 0.0
        running_feat = 0.0
        total_seen = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                with autocast():
                    teacher_logits, teacher_feat = forward_resnet_with_features(teacher_backbone, inputs)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                student_logits, student_feat = forward_resnet_with_features(student_backbone, inputs)
                student_feat_proj = student_projector(student_feat)

            kd_total, kd_soft_loss, hard_loss = kd_loss(
                student_logits,
                teacher_logits,
                targets,
                cfg.temperature,
                cfg.alpha,
                hard_criterion,
            )
            feat_loss = feature_distill_loss(student_feat_proj, teacher_feat)

            loss = kd_total + cfg.feature_beta * feat_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(student_backbone.parameters()) + list(student_projector.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            running_total += loss.item() * inputs.size(0)
            running_kd += kd_total.item() * inputs.size(0)
            running_hard += hard_loss.item() * inputs.size(0)
            running_feat += feat_loss.item() * inputs.size(0)
            total_seen += inputs.size(0)

        scheduler.step()

        elapsed = time.time() - t0
        total_train_time += elapsed

        train_loss = running_total / max(1, total_seen)
        train_kd_loss = running_kd / max(1, total_seen)
        train_hard_loss = running_hard / max(1, total_seen)
        train_feat_loss = running_feat / max(1, total_seen)
        imgs_per_sec = total_seen / max(elapsed, 1e-8)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated(device))

        val_loss, val_acc = evaluate(student_backbone, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_kd_loss": train_kd_loss,
            "train_hard_loss": train_hard_loss,
            "train_feat_loss": train_feat_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "imgs_per_sec": imgs_per_sec,
            "epoch_time": elapsed,
            "peak_mem_gb": peak_mem_gb,
        }
        history.append(row)

        print(
            f"{epoch:>6} {current_lr:>8.5f} {train_loss:>10.4f} {train_kd_loss:>8.4f} "
            f"{train_feat_loss:>8.4f} {val_loss:>9.4f} {val_acc:>7.2f}% "
            f"{imgs_per_sec:>8.0f} {peak_mem_gb:>8.2f} {elapsed:>5.0f}s",
            flush=True,
        )

        improved = val_loss < (best_val_loss - cfg.min_delta)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            # Save backbone only for clean inference / quantization later.
            save_checkpoint(checkpoint_path, student_backbone, optimizer, best_val_loss, best_val_acc, best_epoch, cfg)
        else:
            epochs_no_improve += 1

        save_json(results_json, {
            "config": asdict(cfg),
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "total_train_time_sec": total_train_time,
            "peak_mem_gb": peak_mem_bytes / (1024 ** 3),
            "train_size": len(train_idx),
            "val_size": len(internal_val_idx),
            "benchmark_size": len(benchmark_dataset),
            "classes": num_classes,
            "model": "resnet18_kd_feature",
            "dataset": "TinyImageNet-200",
            "teacher_benchmark_acc": teacher_benchmark_acc,
            "teacher_params_M": teacher_params,
            "student_params_M": student_params,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        })
        save_csv(results_csv, history)

        if epochs_no_improve >= cfg.patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}"
            )
            break

    total_wall_time = time.time() - global_start

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best student checkpoint not found at {checkpoint_path}")

    # Load backbone only for final benchmark, because projector is training-only.
    student_eval = build_student(num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student_eval.load_state_dict(ckpt["model_state_dict"])

    benchmark_loss, benchmark_acc = evaluate(student_eval, benchmark_loader, device, criterion)

    final_summary = {
        "config": asdict(cfg),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "benchmark_loss": benchmark_loss,
        "benchmark_acc": benchmark_acc,
        "total_train_time_sec": total_train_time,
        "total_wall_time_sec": total_wall_time,
        "peak_mem_gb": peak_mem_bytes / (1024 ** 3),
        "train_size": len(train_idx),
        "val_size": len(internal_val_idx),
        "benchmark_size": len(benchmark_dataset),
        "classes": num_classes,
        "model": "resnet18_kd_feature",
        "dataset": "TinyImageNet-200",
        "teacher_benchmark_acc": teacher_benchmark_acc,
        "teacher_params_M": teacher_params,
        "student_params_M": student_params,
        "checkpoint_path": checkpoint_path,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    save_json(results_json, final_summary)

    print(f"\nTeacher benchmark acc : {teacher_benchmark_acc if teacher_benchmark_acc is not None else 'N/A'}")
    print(f"Student best val loss  : {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Student best val acc   : {best_val_acc:.2f}%")
    print(f"Benchmark loss         : {benchmark_loss:.4f}")
    print(f"Benchmark acc          : {benchmark_acc:.2f}%")
    print(f"Total train time       : {total_train_time/3600:.2f} h")
    print(f"Total wall time        : {total_wall_time/3600:.2f} h")
    print(f"Peak GPU memory        : {final_summary['peak_mem_gb']:.2f} GB")
    print(f"Avg imgs/sec           : {np.mean([h['imgs_per_sec'] for h in history]):.0f}")
    print(f"Checkpoint             : {checkpoint_path}")


if __name__ == "__main__":
    main()