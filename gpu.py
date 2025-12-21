import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# -------------------------
# Model (BASIC TRANSFORMER)
# -------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(10, 16)
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


# -------------------------
# Single-process GPU run
# -------------------------
def run_single_process():
    print("\n========== SINGLE PROCESS GPU ==========")
    device = torch.device("cuda:0")

    model = TinyTransformer().to(device)

    x = torch.randn(64, 4, 10)
    y = torch.randn(64, 1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    start = time.time()

    for epoch in range(2):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        print(f"[Single GPU] Epoch {epoch} Loss {loss.item():.4f}")

    elapsed = time.time() - start
    print(f"[Single GPU] Total time: {elapsed:.4f}s")
    return elapsed


# -------------------------
# DDP run (2 processes, SAME GPU)
# -------------------------
def run_ddp():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda:0")  # both ranks use same GPU

    model = TinyTransformer().to(device)
    ddp_model = DDP(model)

    x = torch.randn(64, 4, 10)
    y = torch.randn(64, 1)
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    start = time.time()

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ddp_model(bx), by)
            loss.backward()
            optimizer.step()
        print(f"[Rank {rank}] Epoch {epoch} Loss {loss.item():.4f}")

    elapsed = time.time() - start

    # -------- Parameter sync check (explicit print) --------
    with torch.no_grad():
        local_w = ddp_model.module.fc.weight.clone()

    w_rank0 = local_w.clone()
    dist.broadcast(w_rank0, src=0)

    print(f"\n[Rank {rank}] First 5 weights from rank0: {w_rank0.view(-1)[:5]}")
    print(f"[Rank {rank}] First 5 local weights : {local_w.view(-1)[:5]}")

    diff = torch.norm(w_rank0 - local_w).item()
    print(f"[Rank {rank}] Param diff from rank0: {diff:.6f}")

    return elapsed


# -------------------------
# Main entry
# -------------------------
def main():
    # If NOT launched with torchrun → single-process baseline
    if "RANK" not in os.environ:
        run_single_process()
        print("\nNow run the SAME file with:")
        print("torchrun --nproc_per_node=2 gpu_gl.py")
        return

    # If launched with torchrun → DDP section
    dist.init_process_group(backend="gloo")

    ddp_time = run_ddp()

    if dist.get_rank() == 0:
        print(f"\n[DDP same GPU] Total time: {ddp_time:.4f}s")
        print("Note: No real speedup is expected because both processes share one GPU.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
