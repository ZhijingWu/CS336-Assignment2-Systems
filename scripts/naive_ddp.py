import torch
import torch.nn as nn
import torch.distributed as dist
import os
import torch.multiprocessing as mp


class NaiveDDP(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module

        for param in self.module.parameters():
            dist.broadcast(
                param.data,
                src=0,
            )

    def forward(self, *args, **kwargs):
        return self.module(
            *args,
            **kwargs,
        )

    def sync_gradients(self):
        world_size = dist.get_world_size()

        for param in self.module.parameters():
            if param.grad is None:
                continue

            dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
            )

            param.grad /= world_size


class FlatDDP(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_gradients(self):
        world_size = dist.get_world_size()

        params = [
            p for p in self.module.parameters()
            if p.grad is not None
        ]

        grads = [
            p.grad for p in params
        ]

        flat_grad = torch._utils._unflatten_dense_tensors(
            flat_grad,
            grads,
        )

        dist.all_reduce(
            flat_grad,
            op=dist.ReduceOp.SUM,
        )

        flat_grad /= world_size

        synced_grads = torch._utils._unflatten_dense_tensors(
            flat_grad,
            grads,
        )

        for param, synced_grad in zip(
            params,
            synced_grads,
        ):
            param.grad_copy_(synced_grad)













def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def worker(rank, world_size):
    setup(rank, world_size)

    device = torch.device(
        f"cuda:{rank}"
    )

    torch.manual_seed(
        1234 + rank
    )

    model = nn.Linear(
        4,
        1,
        bias=False,
    ).to(device)

    model = NaiveDDP(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    # 每个 rank 故意用不同数据
    x = torch.randn(
        2,
        4,
        device=device,
    )

    y = torch.randn(
        2,
        1,
        device=device,
    )

    optimizer.zero_grad()

    pred = model(x)

    loss = (
        (pred - y) ** 2
    ).mean()

    loss.backward()

    print(
        f"rank {rank} grad before sync:",
        model.module.weight.grad,
    )

    model.sync_gradients()

    print(
        f"rank {rank} grad after sync:",
        model.module.weight.grad,
    )

    optimizer.step()

    print(
        f"rank {rank} weight after step:",
        model.module.weight.data,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2

    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )