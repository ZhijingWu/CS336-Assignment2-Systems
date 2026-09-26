import torch
import triton
import triton.language as tl


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,

    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,

    NUM_ROWS,
    D,

    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    # 我是第几个 row tile
    row_tile_idx = tl.program_id(0)

    # X 当前 tile
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
            0,
        ),
        block_shape=(
            ROWS_TILE_SIZE,
            D_TILE_SIZE,
        ),
        order=(1, 0),
    )

    # weight 当前 tile
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # output 当前 tile
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
        ),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 每一行一个累加结果
    acc = tl.zeros(
        (ROWS_TILE_SIZE,),
        dtype=tl.float32,
    )

    # 沿着 D 维一块一块扫描
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        x_block = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        weight_block = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # [ROWS_TILE_SIZE, D_TILE_SIZE]
        product = x_block * weight_block[None, :]

        # 每一行沿 D 求和
        acc += tl.sum(
            product,
            axis=1,
        )

        # 移动到下一块 D
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )

        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )

    # 写回显存
    tl.store(
        output_block_ptr,
        acc,
        boundary_check=(0,),
    )


def weighted_sum_triton(x, weight):
    NUM_ROWS, D = x.shape

    output = torch.empty(
        NUM_ROWS,
        device=x.device,
        dtype=torch.float32,
    )

    ROWS_TILE_SIZE = 16
    D_TILE_SIZE = 16

    grid = (
        triton.cdiv(NUM_ROWS, ROWS_TILE_SIZE),
    )

    weighted_sum_fwd[grid](
        x,
        weight,
        output,

        x.stride(0),
        x.stride(1),
        weight.stride(0),
        output.stride(0),

        NUM_ROWS,
        D,

        ROWS_TILE_SIZE=ROWS_TILE_SIZE,
        D_TILE_SIZE=D_TILE_SIZE,
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(
        32,
        64,
        device="cuda",
    )

    weight = torch.randn(
        64,
        device="cuda",
    )

    ref = (
        x * weight
    ).sum(dim=-1)

    out = weighted_sum_triton(
        x,
        weight,
    )

    print("ref shape:", ref.shape)
    print("out shape:", out.shape)

    print(
        "max abs error:",
        (ref - out).abs().max().item(),
    )

    print(
        "mean abs error:",
        (ref - out).abs().mean().item(),
    )