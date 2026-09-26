import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def distributed_demo(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    data = torch.tensor(
        [rank + 1.0],
        device=device,
    )

    print(
        f"rank {rank}, device={device}, "
        f"before all_reduce: {data}"
    )

    dist.all_reduce(
        data,
        op=dist.ReduceOp.SUM,
    )

    print(
        f"rank {rank}, device={device}, "
        f"after all_reduce: {data}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2

    mp.spawn(
        distributed_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )