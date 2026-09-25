import math
import time
import torch


def attention(q, k, v):
    d = q.shape[-1]

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d)

    probs = torch.softmax(scores, dim=-1)

    output = probs @ v
    return output


def benchmark_attention(batch_size, seq_len, d_model, warmup=5, iters=100):
    device = "cuda"

    q = torch.randn(
        batch_size, seq_len, d_model,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size, seq_len, d_model,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size, seq_len, d_model,
        device=device,
        requires_grad=True,
    )

    # warmup
    for _ in range(warmup):
        y = attention(q, k, v)
        loss = y.sum()
        loss.backward()

        q.grad = None
        k.grad = None
        v.grad = None

    torch.cuda.synchronize()

    # 1. forward timing
    forward_times = []

    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        y = attention(q, k, v)

        torch.cuda.synchronize()
        end = time.perf_counter()

        forward_times.append(end - start)

        del y

    # 2. memory before backward
    torch.cuda.empty_cache()

    y = attention(q, k, v)

    memory_before_backward_mib = (
        torch.cuda.memory_allocated() / (1024 ** 2)
    )

    del y
    torch.cuda.empty_cache()

    # 3. backward timing
    backward_times = []

    for _ in range(iters):
        y = attention(q, k, v)
        loss = y.sum()

        torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        end = time.perf_counter()

        backward_times.append(end - start)

        q.grad = None
        k.grad = None
        v.grad = None

    forward_ms = sum(forward_times) / len(forward_times) * 1000
    backward_ms = sum(backward_times) / len(backward_times) * 1000

    print(
        f"seq_len={seq_len}, d={d_model}, "
        f"forward={forward_ms:.3f} ms, "
        f"backward={backward_ms:.3f} ms, "
        f"memory_before_backward={memory_before_backward_mib:.2f} MiB"
    )


if __name__ == "__main__":
    dims = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    for d in dims:
        for seq_len in seq_lens:
            try:
                benchmark_attention(
                    batch_size=8,
                    seq_len=seq_len,
                    d_model=d,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"seq_len={seq_len}, d={d}: OOM")
                torch.cuda.empty_cache()