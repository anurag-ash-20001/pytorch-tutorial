# Distributed Data Parallelism (DDP) – Single GPU & CPU Demo

This repository demonstrates **classic data parallelism** using **PyTorch DistributedDataParallel (DDP)**.

The goal is to **show how data parallelism works in code**, not to achieve speedup.

---

## What is shown

- **2 processes on the same GPU** using `torchrun`
- **2 CPU processes** using `torch.multiprocessing.spawn`
- Full model replication on each process
- Dataset sharding using `DistributedSampler`
- Gradient synchronization using all-reduce

---

## How data parallelism works

1. Multiple processes are launched
2. Each process holds a **full copy of the model**
3. The dataset is split across processes
4. Each process computes gradients on different data
5. Gradients are synchronized, keeping models identical

This is **classic data parallelism**.

---

## Why 2 processes on 1 GPU is useless

- Both processes share the **same GPU**
- GPU computation is **serialized**
- Gradient synchronization adds overhead
- Result: **no real speedup**

DDP guarantees **correctness**, not performance.

---

## Key takeaway

> Data parallelism works correctly, but speedup only happens when each process runs on a different GPU.

---

## Note on ZeRO-Infinity

ZeRO-Infinity is **not implemented** here.  
It is referenced only to understand the **memory and scalability limits** of standard data parallelism demonstrated in this project.


## ▶️ How to Run

### 1️⃣ Single-process baseline (1 GPU, 1 process)
``` python3 gpu.py```
### 2️⃣ DistributedDataParallel (2 processes, SAME GPU)
```torchrun --nproc_per_node=2 gpu.py```
