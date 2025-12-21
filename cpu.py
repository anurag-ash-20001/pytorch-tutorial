import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


# ------------------ Model ------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# ------------------ Distributed Setup ------------------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    print(f"\n🚀 [Rank {rank}] Initialized process group")


def cleanup(rank):
    dist.destroy_process_group()
    print(f"🧹 [Rank {rank}] Destroyed process group\n")


# ------------------ Training Function ------------------
def train_ddp(rank, world_size):
    setup(rank, world_size)

    torch.manual_seed(42 + rank)

    # ---- Model ----
    model = TinyModel()
    ddp_model = DDP(model)

    print(f"[Rank {rank}] Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # ---- Dataset ----
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler
    )

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # ---- Training Loop ----
    for epoch in range(2):
        sampler.set_epoch(epoch)

        print(f"\n📘 [Rank {rank}] Starting Epoch {epoch}")
        epoch_loss = 0.0

        for step, (bx, by) in enumerate(loader):
            optimizer.zero_grad()

            preds = ddp_model(bx)
            loss = loss_fn(preds, by)

            # BACKWARD = gradient computation + synchronization
            loss.backward()

            # DDP synchronizes gradients HERE
            if step == 0:
                print(f"🔄 [Rank {rank}] Gradients synchronized via all-reduce")

            optimizer.step()

            epoch_loss += loss.item()

            print(
                f"[Rank {rank}] Epoch {epoch} | "
                f"Batch {step:02d} | "
                f"Batch Size {bx.size(0)} | "
                f"Loss {loss.item():.4f}"
            )

        avg_loss = epoch_loss / len(loader)
        print(f"✅ [Rank {rank}] Epoch {epoch} completed | Avg Loss {avg_loss:.4f}")

    cleanup(rank)


# ------------------ Entry Point ------------------
if __name__ == "__main__":
    world_size = 2

    print("\n🔥 Launching Distributed Training 🔥")
    print(f"🌍 World Size: {world_size} processes\n")

    mp.spawn(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

    print("\n🏁 Training finished successfully")
