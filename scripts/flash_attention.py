import math
import torch


class FlashAttentionPyTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Bq = 16
        Bk = 16

        scale = 1 / math.sqrt(D)

        O = torch.empty_like(Q)
        L = torch.zeros(
            B,Nq,
            device=Q.device,
            dtype=torch.float32,
        )

        for b in range(B):
            for q_start in range(0, Nq, Bq):
                q_end = q_start + Bq
                Qi = Q[b, q_start:q_end]


                # running
                m = torch.full(
                    (Bq,),
                    float("-inf"),
                    device=Q.device,
                    dtype=torch.float32,
                )

                l = torch.zeros(
                    Bq,
                    device=Q.device,
                    dtype=torch.float32,
                )

                Oi = torch.zeros(
                    Bq,
                    D,
                    device=Q.device,
                    dtype=torch.float32,
                )

                for k_start in range(0, Nk, Bk):
                    k_end = k_start + Bk
                    Kj = K[b, k_start:k_end]
                    Vj = V[b, k_start:k_end]

                    S = (Qi @ Kj.T) * scale

                    m_new = torch.maximum(
                        m,
                        S.max(dim=-1).values
                    )

                    P_tilde = torch.exp(
                        S - m_new[:, None]
                    )

                    alpha = torch.exp(m - m_new)

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
                L[b, q_start:q_end] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError



def naive_attention(Q, K, V):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    P = torch.softmax(S, dim=-1)
    return P @ V


if __name__ == "__main__":
    torch.manual_seed(0)

    Q = torch.randn(1, 32, 16, device="cuda")
    K = torch.randn(1, 32, 16, device="cuda")
    V = torch.randn(1, 32, 16, device="cuda")

    ref = naive_attention(Q, K, V)
    out = FlashAttentionPyTorch.apply(Q, K, V, False)

    print("ref shape:", ref.shape)
    print("out shape:", out.shape)
    print("max abs error:", (ref - out).abs().max().item())
    print("mean abs error:", (ref - out).abs().mean().item())

