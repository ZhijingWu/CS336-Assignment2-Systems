import math
import torch
import triton

from flash_attention import FlashAttentionTriton


def regular_attention(Q, K, V, is_causal=True):
    """
    普通 PyTorch attention。
    注意这里不用 scaled_dot_product_attention，
    因为它有可能底层自动调用 FlashAttention。
    """
    D = Q.shape[-1]

    S = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(D)

    if is_causal:
        Nq = Q.shape[-2]
        Nk = K.shape[-2]

        q_idx = torch.arange(
            Nq,
            device=Q.device,
        )[:, None]

        k_idx = torch.arange(
            Nk,
            device=Q.device,
        )[None, :]

        mask = q_idx >= k_idx

        S = S.masked_fill(
            ~mask,
            float("-inf"),
        )

    P = torch.softmax(S, dim=-1)

    return P @ V


def make_inputs(N, D, dtype):
    Q = torch.randn(
        1,
        N,
        D,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    K = torch.randn(
        1,
        N,
        D,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    V = torch.randn(
        1,
        N,
        D,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    dO = torch.randn_like(Q)

    return Q, K, V, dO


def benchmark_forward(fn, Q, K, V):
    def run():
        fn(Q, K, V)

    return triton.testing.do_bench(
        run,
        warmup=100,
        rep=300,
    )


def benchmark_backward(fn, Q, K, V, dO):
    # 先建一次计算图
    O = fn(Q, K, V)

    def run():
        Q.grad = None
        K.grad = None
        V.grad = None

        O.backward(
            dO,
            retain_graph=True,
        )

    return triton.testing.do_bench(
        run,
        warmup=100,
        rep=300,
    )


def benchmark_end_to_end(fn, Q, K, V, dO):
    def run():
        Q.grad = None
        K.grad = None
        V.grad = None

        O = fn(Q, K, V)

        O.backward(dO)

    return triton.testing.do_bench(
        run,
        warmup=100,
        rep=300,
    )


def run_one(N, D, dtype):
    Q, K, V, dO = make_inputs(
        N,
        D,
        dtype,
    )

    pytorch_fn = lambda Q, K, V: regular_attention(
        Q,
        K,
        V,
        is_causal=True,
    )

    flash_fn = lambda Q, K, V: FlashAttentionTriton.apply(
        Q,
        K,
        V,
        True,
    )

    result = {
        "N": N,
        "D": D,
        "dtype": str(dtype).replace("torch.", ""),
    }

    try:
        result["torch_fwd"] = benchmark_forward(
            pytorch_fn,
            Q,
            K,
            V,
        )

        result["flash_fwd"] = benchmark_forward(
            flash_fn,
            Q,
            K,
            V,
        )

        result["torch_bwd"] = benchmark_backward(
            pytorch_fn,
            Q,
            K,
            V,
            dO,
        )

        result["flash_bwd"] = benchmark_backward(
            flash_fn,
            Q,
            K,
            V,
            dO,
        )

        result["torch_e2e"] = benchmark_end_to_end(
            pytorch_fn,
            Q,
            K,
            V,
            dO,
        )

        result["flash_e2e"] = benchmark_end_to_end(
            flash_fn,
            Q,
            K,
            V,
            dO,
        )

        return result

    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

        result["OOM"] = True

        return result


def print_result(r):
    if r.get("OOM", False):
        print(
            f"N={r['N']:5d} "
            f"D={r['D']:3d} "
            f"dtype={r['dtype']:8s} "
            f"OOM"
        )
        return

    print(
        f"N={r['N']:5d} "
        f"D={r['D']:3d} "
        f"dtype={r['dtype']:8s} | "
        f"fwd: torch={r['torch_fwd']:8.3f} ms "
        f"flash={r['flash_fwd']:8.3f} ms | "
        f"bwd: torch={r['torch_bwd']:8.3f} ms "
        f"flash={r['flash_bwd']:8.3f} ms | "
        f"e2e: torch={r['torch_e2e']:8.3f} ms "
        f"flash={r['flash_e2e']:8.3f} ms"
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    # 先小范围跑通。
    # 跑通以后再往 8192 / 16384 扩。
    sequence_lengths = [
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]

    dimensions = [
        16,
        32,
        64,
        128,
    ]

    dtypes = [
        torch.bfloat16,
        torch.float32,
    ]

    for dtype in dtypes:
        for D in dimensions:
            for N in sequence_lengths:

                print(
                    f"\nBenchmarking "
                    f"N={N}, D={D}, dtype={dtype}"
                )

                result = run_one(
                    N,
                    D,
                    dtype,
                )

                print_result(result)

                torch.cuda.empty_cache()