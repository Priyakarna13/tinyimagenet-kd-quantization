#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tinyimg_int8_ptq
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 0-06:00:00
#SBATCH -o /data/pjiang18/SLP/ModelCompression/logs/int8_ptq_%j.out
#SBATCH -e /data/pjiang18/SLP/ModelCompression/logs/int8_ptq_%j.err
#SBATCH --export=NONE

module load mamba/latest
source activate /data/pjiang18/SLP/envs/llm-bench

cd /data/pjiang18/SLP/ModelCompression

python eval_int8_ptq_student.py \
  --student_ckpt /data/pjiang18/SLP/ModelCompression/checkpoints/tinyimagenet_resnet18_student_best.pth \
  --teacher_results /data/pjiang18/SLP/ModelCompression/results/tinyimagenet_teacher_results.json \
  --results_file /data/pjiang18/SLP/ModelCompression/results/tinyimagenet_int8_ptq_results.json \
  --seed 42 \
  --batch_size_eval 128 \
  --batch_size_tp 32 \
  --batch_size_latency 1 \
  --calibration_batches 32 \
  --latency_warmup_batches 20 \
  --latency_measure_batches 100 \
  --num_workers 4 \
  --num_threads 8