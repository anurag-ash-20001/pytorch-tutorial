# 🚀 PyTorch GPU Setup Guide (RTX 4060 – Windows & Linux)

This repository provides **complete step-by-step instructions** to install **PyTorch with CUDA support** and run it using an **NVIDIA RTX 4060 GPU** on **Windows** and **Linux**.

✅ No manual CUDA Toolkit installation required  
✅ Fully compatible with RTX 40-series GPUs  
✅ Beginner-friendly and VS Code ready  

---

## 📋 Requirements

### Hardware
- NVIDIA RTX 4060 (Laptop or Desktop)

### Software
- Python 3.9 – 3.12
- NVIDIA GPU Driver (latest recommended)
- Git (optional)
- VS Code (recommended)

---

## 🔍 Verify NVIDIA Driver (Windows & Linux)

Run:
```bash
nvidia-smi
```

You should see your RTX 4060 listed.
If not, install or update your NVIDIA driver first.

## 🐍 Create a Virtual Environment (Recommended)
-Windows

```
python -m venv venv
venv\Scripts\activate
```

-Linux

```
python3 -m venv venv
source venv/bin/activate
```

You should see (venv) in your terminal.
📦 Upgrade pip

```
python -m pip install --upgrade pip
```

##📦 Install Required Dependencies
Install NumPy
```
pip install numpy
```

##⚡ Install PyTorch with CUDA (RTX 4060 Compatible)

⚠️ IMPORTANT
Do NOT install CUDA Toolkit manually.
PyTorch includes its own CUDA runtime.
Windows & Linux (CUDA 12.1 – Recommended)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
This installs:

    PyTorch with CUDA support

    torchvision

    torchaudio

✅ Verify GPU Installation

Run Python:

python

Then execute:

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

Expected Output

PyTorch version: 2.x.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4060

If CUDA available is True, your GPU is working correctly 🚀
🧪 Test Script (GPU Matrix Multiplication)

Create a file called test_gpu.py:

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

x = torch.rand(1000, 1000, device=device)
y = torch.rand(1000, 1000, device=device)

z = x @ y
print("Matrix multiplication completed on", device)

Run it:

python test_gpu.py

🧠 VS Code Setup (Important)

    Open Command Palette: Ctrl + Shift + P

    Select Python: Select Interpreter

    Choose:

        Windows: venv\Scripts\python.exe

        Linux: venv/bin/python

This ensures VS Code uses the CUDA-enabled PyTorch.
❌ Common Issues & Fixes
Torch not compiled with CUDA enabled

Cause: CPU-only PyTorch installed

Fix:

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

torch.cuda.is_available() returns False

    Ensure NVIDIA driver is installed

    Ensure correct Python interpreter is selected

    Ensure CUDA-enabled PyTorch is installed

    Ensure virtual environment is activated

📌 Notes

    RTX 4060 works best with CUDA 12.x

    No need to install CUDA Toolkit manually

    Always use a virtual environment

    Works on both Windows and Linux

🎯 Next Steps

    Train neural networks on GPU

    Benchmark CPU vs GPU performance

    Run deep learning models (CNNs, Transformers)

📜 License

MIT License
