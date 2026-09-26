import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,

    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,

    N_QUERIES,
    N_KEYS,
    scale,

    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),
        block_shape=(
            Q_TILE_SIZE,
            D,
        ),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(
            K_TILE_SIZE,
            D,
        ),
        order=(1,0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(
            K_TILE_SIZE,
            D,
        ),
        order=(1,0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),
        block_shape=(
            Q_TILE_SIZE,
            D,
        ),
        order=(1,0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
        ),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr)

    m = tl.full(
        (Q_TILE_SIZE,),
        float("-inf"),
        tl.float32,
    )

    l = tl.zeros(
        (Q_TILE_SIZE,),
        tl.float32,
    )

    Oi = tl.zeros(
        (Q_TILE_SIZE, D),
        tl.float32,
    )

    q_offsets = (
        query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    )

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        k_offsets = (
            j * K_TILE_SIZE
            + tl.arange(0, K_TILE_SIZE)
        )

        S = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            mask = (
                q_offsets[:, None]
                >=
                k_offsets[None, :]
            )

            S += tl.where(
                mask,
                0.0,
                -1e6
            )

        m_new = tl.maximum(
            m,
            tl.max(S, axis=1),
        )

        P_tilde = tl.exp(
            S - m_new[:, None]
        )

        alpha = tl.exp(
            m - m_new
        )

        l_new = (
            alpha * l
            + tl.sum(P_tilde, axis=1)
        )

        Oi = (
            alpha[:, None] * Oi
            + tl.dot(
                P_tilde.to(Vj.dtype),
                Vj,
            )
        )

        m = m_new
        l = l_new

        K_block_ptr = K_block_ptr.advance(
            (K_TILE_SIZE, 0)
        )

        V_block_ptr = V_block_ptr.advance(
            (K_TILE_SIZE, 0)
        )

    Oi = Oi / l[:, None]
    Li = m + tl.log(l) 

    tl.store(
        O_block_ptr,
        Oi.to(O_block_ptr.type.element_ty),
        boundary_check=(0,1),
    )

    tl.store(
        L_block_ptr,
        Li,
        boundary_check=(0,),
    )


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        _, Nk, _ = K.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        scale = 1.0 / math.sqrt(D)

        O = torch.empty_like(Q)

        L = torch.empty(
            B,
            Nq,
            device=Q.device,
            dtype=torch.float32,
        )

        grid = (
            triton.cdiv(Nq, Q_TILE_SIZE),
            B,
        )

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,

            Q.stride(0),
            Q.stride(1),
            Q.stride(2),

            K.stride(0),
            K.stride(1),
            K.stride(2),

            V.stride(0),
            V.stride(1),
            V.stride(2),

            O.stride(0),
            O.stride(1),
            O.stride(2),

            L.stride(0),
            L.stride(1),

            Nq,
            Nk,
            scale,

            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)

        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors

        dQ, dK, dV = flash_backward_pytorch(
            Q,
            K,
            V,
            O,
            grad_out,
            L,
            ctx.is_causal,
        )

        return dQ, dK, dV, None
    

class FlashAttentionPyTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        Nk = K.shape[1]

        device = Q.device

        Bq = 16
        Bk = 16

        scale = 1.0 / math.sqrt(D)

        O = torch.empty_like(Q)

        L = torch.empty(
            B,
            Nq,
            device=device,
            dtype=torch.float32,
        )

        for b in range(B):
            for q_start in range(0, Nq, Bq):
                q_end = q_start + Bq
                Qi = Q[b, q_start:q_end]  # [Bq, D]

                m = torch.full(
                    (Bq,),
                    float("-inf"),
                    device=device,
                    dtype=torch.float32,
                )

                l = torch.zeros(
                    Bq,
                    device=device,
                    dtype=torch.float32,
                )

                Oi = torch.zeros(
                    Bq,
                    D,
                    device=device,
                    dtype=torch.float32,
                )

                for k_start in range(0, Nk, Bk):
                    k_end = k_start + Bk

                    Kj = K[b, k_start:k_end]  # [Bk, D]
                    Vj = V[b, k_start:k_end]  # [Bk, D]

                    S = (Qi @ Kj.T) * scale  # [Bq, Bk]

                    m_new = torch.maximum(
                        m,
                        S.max(dim=-1).values,
                    )

                    P_tilde = torch.exp(
                        S - m_new[:, None]
                    )

                    alpha = torch.exp(
                        m - m_new
                    )

                    l_new = (
                        alpha * l
                        + P_tilde.sum(dim=-1)
                    )

                    Oi = (
                        alpha[:, None] * Oi
                        + P_tilde @ Vj
                    )

                    m = m_new
                    l = l_new

                Oi = Oi / l[:, None]

                O[b, q_start:q_end] = Oi.to(Q.dtype)

                L[b, q_start:q_end] = (
                    m + torch.log(l)
                )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors

        dQ, dK, dV = flash_backward_pytorch(
            Q,
            K,
            V,
            O,
            grad_out,
            L,
            ctx.is_causal,
        )

        return dQ, dK, dV, None


def flash_backward_pytorch(
    Q, K, V, O, dO, L,
    is_causal=False,
):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    S = (
        Q @ K.transpose(-2, -1)
    ) * scale

    if is_causal:
        Nq = Q.shape[-2]
        Nk = K.shape[-2]

        q_idx = torch.arange(
            Nq,
            device=Q.device,
        )[:, None]

        k_idx = torch.arange(
            Nk,
            device=K.device,
        )[None, :]

        mask = q_idx >= k_idx

        S = S.masked_fill(
            ~mask,
            float("-inf"),
        )

    P = torch.exp(
        S - L[:, :, None]
    ).to(Q.dtype)

    D = (O * dO).sum(
        dim=-1,
        dtype=torch.float32,
    )

    dV = (
        P.transpose(-2, -1)
        @ dO
    )

    dP = (
        dO
        @ V.transpose(-2, -1)
    )

    dS = (
        P
        * (
            dP
            - D[:, :, None].to(dP.dtype)
        )
    )

    dQ = (
        dS @ K
    ) * scale

    dK = (
        dS.transpose(-2, -1)
        @ Q
    ) * scale

    return dQ, dK, dV


flash_backward_pytorch = torch.compile(
    flash_backward_pytorch
)


def naive_attention(Q, K, V):
    S = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(Q.shape[-1])

    P = torch.softmax(
        S,
        dim=-1,
    )

    return P @ V


if __name__ == "__main__":
    torch.manual_seed(0)

    Q = torch.randn(
        1, 32, 16,
        device="cuda",
        requires_grad=True,
    )

    K = torch.randn(
        1, 32, 16,
        device="cuda",
        requires_grad=True,
    )

    V = torch.randn(
        1, 32, 16,
        device="cuda",
        requires_grad=True,
    )

    ref = naive_attention(Q, K, V)

    out = FlashAttentionTriton.apply(
        Q,
        K,
        V,
        False,
    )

    print("ref shape:", ref.shape)
    print("out shape:", out.shape)

    print(
        "max abs error:",
        (ref - out).abs().max().item()
    )

    print(
        "mean abs error:",
        (ref - out).abs().mean().item()
    )

    print("ref[0, 0, :5] =", ref[0, 0, :5])
    print("out[0, 0, :5] =", out[0, 0, :5])

    print("ref[0, 16, :5] =", ref[0, 16, :5])
    print("out[0, 16, :5] =", out[0, 16, :5])