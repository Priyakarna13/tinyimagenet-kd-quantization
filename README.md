# TinyImageNet Model Compression Pipeline

A complete deep learning model compression pipeline using Knowledge Distillation and Post-Training Quantization to reduce model size and inference latency while retaining accuracy.

---

## 🚀 Results Summary

| Model | Accuracy (%) | Size (MB) | Latency (ms/img) | Throughput (img/s) |
|---|---|---|---|---|
| Teacher (ResNet50) | 82.19 | ~90+ | — | — |
| KD Student (ResNet18) | 70.70 | 43.10 | 11.80 | 92.18 |
| INT8 Student | 70.37 | 10.89 | 5.43 | 217.78 |

**Key Observations**
- KD reduces model size ~2x with acceptable accuracy drop
- INT8 quantization reduces size ~4x and improves speed ~2x
- Accuracy drop from KD → INT8 is minimal (0.33%)

---

## 📁 Project Structure

```
.
├── train_teacher_tinyimagenet.py
├── train_student_kd_tinyimagenet.py
├── eval_int8_ptq_student.py
├── run_teacher_tinyimagenet.sh
├── run_student_kd_tinyimagenet.sh
├── run_int8_ptq_student.sh
├── results/
├── logs/
├── data/
├── checkpoints/
└── tinyimagenet_figures/
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/tinyimagenet-kd-int8-compression.git
cd tinyimagenet-kd-int8-compression
```

### 2. Prepare Required Folders

```bash
mkdir -p data checkpoints tinyimagenet_figures
```

### 3. Add Dataset

Download the [TinyImageNet dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and place it inside `data/tiny-imagenet-200/` with the following structure:

```
data/
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    └── ...
```

### 4. Train Teacher Model

```bash
bash run_teacher_tinyimagenet.sh
```

Generates: `checkpoints/tinyimagenet_resnet50_teacher_best.pth`

### 5. Train KD Student Model

```bash
bash run_student_kd_tinyimagenet.sh
```

Generates: `checkpoints/tinyimagenet_resnet18_student_best.pth`

### 6. Run INT8 Quantization (PTQ)

```bash
bash run_int8_ptq_student.sh
```

Evaluates both the float student and INT8 quantized student.

---

## 📊 Outputs

| Directory | Contents |
|---|---|
| `results/` | Training logs, accuracy metrics, quantization results |
| `logs/` | Run logs |
| `tinyimagenet_figures/` | Training and evaluation plots |

---

## 🧠 Key Techniques

- **Transfer Learning** — Pretrained ResNet backbone
- **Knowledge Distillation** — Soft targets combined with hard labels
- **Post-Training Quantization** — INT8 with FBGEMM backend
- **Benchmarking** — Latency, throughput, and model size comparisons

---

## ⚠️ Notes

- Dataset is not included due to size constraints
- Model checkpoints (`.pth`) are not included due to GitHub file limits
- Ensure the correct folder structure is in place before running any scripts

---

## 📌 Conclusion

This project demonstrates an effective pipeline to compress deep learning models, maintain reasonable accuracy, and improve deployment efficiency — making it suitable for resource-constrained production environments.

---

## 👤 Author

**Priyakarna**
