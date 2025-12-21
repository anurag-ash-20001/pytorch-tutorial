#  data parallelism Implementation + ZERO infinty(theory)

## Part A: Read the paper “ZeRO‑Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning”
## https://arxiv.org/pdf/2104.07857 
## https://arxiv.org/abs/1910.02054v3

# Single-GPU Data Parallelism with PyTorch DistributedDataParallel (DDP)

This repository demonstrates **classic data parallelism** using **PyTorch DistributedDataParallel (DDP)** in a **single-GPU environment**, where **two processes target the same GPU**.

The goal of this project is **not performance**, but to **show the data-parallel training pattern in code** and to **explicitly demonstrate why running multiple DDP processes on the same GPU provides no real speedup**.

---

## 🚀 What This Project Demonstrates

- How **DistributedDataParallel (DDP)** works internally
- How **data parallelism is achieved** (same model, different data)
- Why **2 processes on 1 GPU is inefficient**
- How **gradient synchronization keeps models identical**
- How this setup relates to **classic data parallelism** discussed in research papers such as **ZeRO-Infinity**

---

## 📁 Files


---

## 🧠 Model Used

A **basic Transformer encoder** is used to keep the focus on distributed training mechanics rather than model complexity.

- Linear embedding
- One Transformer encoder layer
- Mean pooling
- Final linear output layer

---

## ▶️ How to Run

### 1️⃣ Single-process baseline (1 GPU, 1 process)
``` python3 gpu.py```
### 2️⃣ DistributedDataParallel (2 processes, SAME GPU)
```torchrun --nproc_per_node=2 gpu.py```
