import os
import json
import time
import random
import csv
import argparse
import tempfile
from dataclasses import asdict, dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.models.quantization as qmodels
import torch.ao.quantization as aoq
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image

SAVE_DIR = "/data/pjiang18/SLP/ModelCompression"
DATA_DIR = f"{SAVE_DIR}/data/tiny-imagenet-200"
CHECKPOINT_DIR = f"{SAVE_DIR}/checkpoints"
RESULTS_DIR = f"{SAVE_DIR}/results"
LOG_DIR = f"{SAVE_DIR}/logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_STUDENT_CKPT = f"{CHECKPOINT_DIR}/tinyimagenet_resnet18_student_best.pth"
DEFAULT_TEACHER_RESULTS = f"{RESULTS_DIR}/tinyimagenet_teacher_results.json"
DEFAULT_RESULTS_FILE = f"{RESULTS_DIR}/tinyimagenet_int8_ptq_results.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_state_dict_flexible(model: nn.Module, checkpoint_path: str, strict: bool = True) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        return ckpt
    model.load_state_dict(ckpt, strict=strict)
    return {"model_state_dict": ckpt}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


@torch.inference_mode()
def benchmark_throughput(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    start = time.perf_counter()
    total = 0

    for inputs, _ in loader:
        _ = model(inputs)
        total += inputs.size(0)

    elapsed = time.perf_counter() - start
    return total / max(elapsed, 1e-8)


@torch.inference_mode()
def benchmark_latency_ms(
    model: nn.Module,
    loader: DataLoader,
    warmup_batches: int = 10,
    measure_batches: int = 100,
) -> Tuple[float, float]:
    model.eval()
    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            inputs, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, _ = next(iterator)
        _ = model(inputs)

    latencies = []
    for _ in range(measure_batches):
        try:
            inputs, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, _ = next(iterator)
        start = time.perf_counter()
        _ = model(inputs)
        latencies.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(latencies)), float(np.std(latencies))


def model_state_dict_size_mb(model: nn.Module) -> float:
    fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=RESULTS_DIR)
    os.close(fd)
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return size_mb


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


def build_float_student(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_quantizable_student(num_classes: int) -> nn.Module:
    model = qmodels.resnet18(weights=None, quantize=False, backend="fbgemm", num_classes=num_classes)
    return model


@dataclass
class PTQConfig:
    student_ckpt: str = DEFAULT_STUDENT_CKPT
    teacher_results: str = DEFAULT_TEACHER_RESULTS
    results_file: str = DEFAULT_RESULTS_FILE
    seed: int = 42
    batch_size_eval: int = 128
    batch_size_tp: int = 32
    batch_size_latency: int = 1
    calibration_batches: int = 32
    latency_warmup_batches: int = 20
    latency_measure_batches: int = 100
    num_workers: int = 4
    download: bool = False
    num_threads: int = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt", type=str, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher_results", type=str, default=DEFAULT_TEACHER_RESULTS)
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--batch_size_tp", type=int, default=32)
    parser.add_argument("--batch_size_latency", type=int, default=1)
    parser.add_argument("--calibration_batches", type=int, default=32)
    parser.add_argument("--latency_warmup_batches", type=int, default=20)
    parser.add_argument("--latency_measure_batches", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    cfg = PTQConfig(
        student_ckpt=args.student_ckpt,
        teacher_results=args.teacher_results,
        results_file=args.results_file,
        seed=args.seed,
        batch_size_eval=args.batch_size_eval,
        batch_size_tp=args.batch_size_tp,
        batch_size_latency=args.batch_size_latency,
        calibration_batches=args.calibration_batches,
        latency_warmup_batches=args.latency_warmup_batches,
        latency_measure_batches=args.latency_measure_batches,
        num_workers=args.num_workers,
        download=args.download,
        num_threads=args.num_threads,
    )

    set_seed(cfg.seed)

    torch.backends.quantized.engine = "fbgemm"
    torch.set_num_threads(cfg.num_threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    print(f"Quantized engine: {torch.backends.quantized.engine}")
    print(f"CPU threads: {cfg.num_threads}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_root = os.path.join(DATA_DIR, "train")
    benchmark_root = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not os.path.isdir(benchmark_root):
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_root}")
    if not os.path.exists(cfg.student_ckpt):
        raise FileNotFoundError(f"Student checkpoint not found: {cfg.student_ckpt}")

    base_train = torchvision.datasets.ImageFolder(train_root, transform=None)
    train_labels = [label for _, label in base_train.samples]
    train_idx, _ = stratified_split(train_labels, val_ratio=0.1, seed=cfg.seed)

    train_eval_dataset = torchvision.datasets.ImageFolder(train_root, transform=train_eval_transform)
    benchmark_dataset = TinyImageNetBenchmarkDataset(
        benchmark_root,
        class_to_idx=train_eval_dataset.class_to_idx,
        transform=eval_transform,
    )

    calib_subset = Subset(train_eval_dataset, train_idx)

    calib_loader = DataLoader(
        calib_subset,
        batch_size=cfg.batch_size_tp,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    benchmark_loader_acc = DataLoader(
        benchmark_dataset,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    benchmark_loader_tp = DataLoader(
        benchmark_dataset,
        batch_size=cfg.batch_size_tp,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    latency_subset_size = min(len(benchmark_dataset), max(256, cfg.latency_measure_batches))
    latency_subset = Subset(benchmark_dataset, list(range(latency_subset_size)))
    latency_loader = DataLoader(
        latency_subset,
        batch_size=cfg.batch_size_latency,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    num_classes = len(train_eval_dataset.classes)
    print(f"Classes: {num_classes}")
    print(f"Train calibration samples: {len(calib_subset):,}")
    print(f"Benchmark samples: {len(benchmark_dataset):,}")

    float_student = build_float_student(num_classes=num_classes)
    load_state_dict_flexible(float_student, cfg.student_ckpt, strict=True)

    teacher_benchmark_acc = None
    if os.path.exists(cfg.teacher_results):
        try:
            with open(cfg.teacher_results, "r") as f:
                teacher_results = json.load(f)
            teacher_benchmark_acc = teacher_results.get("benchmark_acc")
        except Exception:
            teacher_benchmark_acc = None

    float_student.eval()

    criterion = nn.CrossEntropyLoss()

    # Float benchmark on CPU for reference.
    float_loss, float_acc = evaluate(float_student, benchmark_loader_acc, criterion)
    float_tp = benchmark_throughput(float_student, benchmark_loader_tp)
    float_lat_ms, float_lat_std = benchmark_latency_ms(
        float_student,
        latency_loader,
        warmup_batches=cfg.latency_warmup_batches,
        measure_batches=cfg.latency_measure_batches,
    )
    float_size_mb = model_state_dict_size_mb(float_student)

    # Build quantizable student from the same float checkpoint.
    quant_student = build_quantizable_student(num_classes=num_classes)
    load_state_dict_flexible(quant_student, cfg.student_ckpt, strict=False)

    quant_student.eval()
    quant_student.fuse_model()
    quant_student.qconfig = aoq.get_default_qconfig("fbgemm")

    aoq.prepare(quant_student, inplace=True)

    # Calibration pass.
    with torch.inference_mode():
        for i, (inputs, _) in enumerate(calib_loader):
            _ = quant_student(inputs)
            if i + 1 >= cfg.calibration_batches:
                break

    aoq.convert(quant_student, inplace=True)
    quant_student.eval()

    quant_loss, quant_acc = evaluate(quant_student, benchmark_loader_acc, criterion)
    quant_tp = benchmark_throughput(quant_student, benchmark_loader_tp)
    quant_lat_ms, quant_lat_std = benchmark_latency_ms(
        quant_student,
        latency_loader,
        warmup_batches=cfg.latency_warmup_batches,
        measure_batches=cfg.latency_measure_batches,
    )
    quant_size_mb = model_state_dict_size_mb(quant_student)

    compression_ratio = float_size_mb / max(quant_size_mb, 1e-8)

    results = {
        "config": asdict(cfg),
        "teacher_benchmark_acc": teacher_benchmark_acc,
        "float_student_loss": float_loss,
        "float_student_acc": float_acc,
        "float_student_latency_ms": float_lat_ms,
        "float_student_latency_std_ms": float_lat_std,
        "float_student_throughput_img_s": float_tp,
        "float_student_size_mb": float_size_mb,
        "quant_student_loss": quant_loss,
        "quant_student_acc": quant_acc,
        "quant_student_latency_ms": quant_lat_ms,
        "quant_student_latency_std_ms": quant_lat_std,
        "quant_student_throughput_img_s": quant_tp,
        "quant_student_size_mb": quant_size_mb,
        "compression_ratio": compression_ratio,
        "num_classes": num_classes,
        "benchmark_size": len(benchmark_dataset),
        "device": "cpu",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "quantized_engine": torch.backends.quantized.engine,
    }

    save_json(cfg.results_file, results)

    csv_path = os.path.join(RESULTS_DIR, "tinyimagenet_int8_ptq_results.csv")
    save_csv(csv_path, [results])

    print("\n=== PTQ Evaluation Summary ===")
    print(f"Teacher benchmark acc   : {teacher_benchmark_acc if teacher_benchmark_acc is not None else 'N/A'}")
    print(f"Float student acc       : {float_acc:.2f}%")
    print(f"Float student loss      : {float_loss:.4f}")
    print(f"Float student latency   : {float_lat_ms:.3f} ms/img")
    print(f"Float student throughput: {float_tp:.2f} img/s")
    print(f"Float student size      : {float_size_mb:.2f} MB")
    print(f"INT8 student acc        : {quant_acc:.2f}%")
    print(f"INT8 student loss       : {quant_loss:.4f}")
    print(f"INT8 student latency    : {quant_lat_ms:.3f} ms/img")
    print(f"INT8 student throughput : {quant_tp:.2f} img/s")
    print(f"INT8 student size       : {quant_size_mb:.2f} MB")
    print(f"Compression ratio       : {compression_ratio:.2f}x")
    print(f"\nSaved JSON: {cfg.results_file}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()