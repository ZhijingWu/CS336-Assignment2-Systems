import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import RotaryEmbedding, TransformerBlock


def run_segment(x, block, num_blocks):
    for _ in range(num_blocks):
        x = block(x)
    return x


def forward_with_checkpointing(block, x, num_blocks, block_size):
    for start in range(0, num_blocks, block_size):
        current_size = min(block_size, num_blocks - start)

        x = checkpoint(
            lambda x, current_size=current_size: run_segment(
                x, block, current_size
            ),
            x,
            use_reentrant=False,
        )

    return x


def run_experiment(
    num_blocks: int,
    checkpoint_block_size: int,
    batch_size: int,
    seq_len: int,
):
    d_model = 2560
    d_ff = 10240
    num_heads = 16

    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        positional_encoder=RotaryEmbedding(
            dim=d_model // num_heads,
            context_length=seq_len,
        ),
    ).cuda()

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device="cuda",
        requires_grad=True,
    )

    torch.cuda.reset_peak_memory_stats()

    y = forward_with_checkpointing(
        block=block,
        x=x,
        num_blocks=num_blocks,
        block_size=checkpoint_block_size,
    )

    loss = y.sum()
    loss.backward()

    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(
        f"num_blocks={num_blocks}, "
        f"checkpoint_block_size={checkpoint_block_size}, "
        f"peak_memory={peak_memory_mib:.2f} MiB"
    )


if __name__ == "__main__":
    run_experiment(
        num_blocks=4,
        checkpoint_block_size=2,
        batch_size=1,
        seq_len=128,
    )