from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist


class FSDP(nn.Module):

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # 真正被包装的完整模型
        self.module = module

        # 当前进程编号
        self.rank = dist.get_rank()

        # 一共有多少个 rank
        self.world_size = dist.get_world_size()

        # None -> fp32 compute
        # torch.float16 -> fp16 communication / compute
        self.compute_dtype = compute_dtype

        # name -> sharding metadata
        self._sharded_params: dict[str, dict] = {}

        # Parameter id -> metadata
        # 后面 hook 查参数会方便一些
        self._param_info: dict[int, dict] = {}

        # 保存异步通信
        self._pending_grad_sync = []

        # 先真正把 Linear / Embedding weight 切开
        self._shard_parameters()

        # 再安装 forward / backward hooks
        self._register_hooks()

    # ============================================================
    # 1. 初始化时切分参数
    # ============================================================

    def _shard_parameters(self):
        from cs336_basics.model import Embedding, Linear

        # named_modules:
        #
        # embedding
        # norm1
        # linear1
        # ...
        for module_name, submodule in self.module.named_modules():

            # 测试明确规定：
            # 只有 Linear / Embedding 做 FSDP
            if not isinstance(submodule, (Linear, Embedding)):
                continue

            # 当前作业中的 Linear / Embedding
            # 真正需要 shard 的就是 weight
            param = submodule.weight

            # 原始完整 shape
            original_shape = param.data.shape

            # 原始元素数量
            original_numel = param.data.numel()

            # 先 flatten 成一维
            flat = param.data.reshape(-1)

            # 每个 rank 保存多少元素
            #
            # ceil(numel / world_size)
            shard_size = (
                original_numel
                + self.world_size
                - 1
            ) // self.world_size

            # pad 后的总长度
            padded_numel = (
                shard_size
                * self.world_size
            )

            # 如果不能整除 world_size
            # 就在后面补 0
            if flat.numel() < padded_numel:
                padded = torch.zeros(
                    padded_numel,
                    device=flat.device,
                    dtype=flat.dtype,
                )

                padded[:original_numel] = flat
                flat = padded

            # 当前 rank 应该保存哪一段
            start = self.rank * shard_size
            end = start + shard_size

            local_shard = (
                flat[start:end]
                .clone()
                .to(torch.float32)
            )

            # 记录全名
            if module_name:
                full_name = f"{module_name}.weight"
            else:
                full_name = "weight"

            info = {
                "name": full_name,
                "module": submodule,
                "param": param,
                "original_shape": original_shape,
                "original_numel": original_numel,
                "shard_size": shard_size,
                "padded_numel": padded_numel,

                # forward/backward gather 时，
                # 暂存真正的 fp32 master shard
                "local_master": local_shard,
            }

            self._sharded_params[full_name] = info
            self._param_info[id(param)] = info

            # 非常关键：
            #
            # 从现在开始 param.data 不再是完整 weight，
            # 而只是当前 rank 的 local shard。
            #
            # optimizer 之后看到的也是这个 shard。
            param.data = local_shard

            param.grad = None

    # ============================================================
    # 2. local shard -> full weight
    # ============================================================

    def _all_gather_weight(self, info):
        param = info["param"]

        # 正常情况下 param.data 就是本 rank 的 fp32 master shard
        local_master = param.data

        # 保存下来，因为一会儿 param.data 会被完整 weight 替换
        info["local_master"] = local_master

        # mixed precision:
        #
        # master weight 保持 fp32，
        # 但是通信前转成 compute_dtype
        if self.compute_dtype is None:
            local_for_compute = local_master
        else:
            local_for_compute = local_master.to(
                self.compute_dtype
            )

        # 每个 rank 准备一个位置，
        # 接收其他 rank 的 shard
        gathered = [
            torch.empty_like(local_for_compute)
            for _ in range(self.world_size)
        ]

        # rank0:
        #   [shard0, shard1]
        #
        # rank1:
        #   [shard0, shard1]
        dist.all_gather(
            gathered,
            local_for_compute,
        )

        # 拼成完整 flatten weight
        full_flat = torch.cat(
            gathered,
            dim=0,
        )

        # 去掉 padding
        full_flat = full_flat[
            : info["original_numel"]
        ]

        # 恢复原始矩阵 shape
        full_weight = full_flat.reshape(
            info["original_shape"]
        )

        # 临时让 module 看见完整 weight
        param.data = full_weight

    # ============================================================
    # 3. full weight -> local shard
    # ============================================================

    def _restore_local_shard(self, info):
        param = info["param"]

        # 恢复长期保存的 fp32 master shard
        param.data = info["local_master"]

    # ============================================================
    # 4. 注册 hooks
    # ============================================================

    def _register_hooks(self):

        for info in self._sharded_params.values():

            submodule = info["module"]
            param = info["param"]

            # -------------------------
            # forward 前
            # -------------------------

            def make_forward_pre_hook(current_info):

                def forward_pre_hook(
                    module,
                    inputs,
                ):
                    # forward 需要完整权重
                    self._all_gather_weight(
                        current_info
                    )

                return forward_pre_hook

            # -------------------------
            # forward 后
            # -------------------------

            def make_forward_post_hook(current_info):

                def forward_post_hook(
                    module,
                    inputs,
                    output,
                ):
                    # forward 已经结束
                    # 不需要继续占完整 weight 的显存
                    self._restore_local_shard(
                        current_info
                    )

                return forward_post_hook

            # -------------------------
            # backward 前
            # -------------------------

            def make_backward_pre_hook(current_info):

                def backward_pre_hook(
                    module,
                    grad_output,
                ):
                    # backward 也需要完整 weight
                    #
                    # 特别是 Linear 计算 grad_input 时
                    # 需要完整 weight
                    self._all_gather_weight(
                        current_info
                    )

                return backward_pre_hook

            submodule.register_forward_pre_hook(
                make_forward_pre_hook(info)
            )

            submodule.register_forward_hook(
                make_forward_post_hook(info)
            )

            submodule.register_full_backward_pre_hook(
                make_backward_pre_hook(info)
            )

            # 只有需要梯度的参数注册 grad hook
            if param.requires_grad:

                def make_grad_hook(current_info):

                    def grad_hook(param):

                        if param.grad is None:
                            return

                        # 此时 param.grad 是完整 weight 的 local gradient
                        full_grad = (
                            param.grad
                            .detach()
                            .reshape(-1)
                            .to(torch.float32)
                        )

                        # padding 到：
                        #
                        # shard_size * world_size
                        padded_numel = current_info[
                            "padded_numel"
                        ]

                        if full_grad.numel() < padded_numel:
                            padded_grad = torch.zeros(
                                padded_numel,
                                device=full_grad.device,
                                dtype=torch.float32,
                            )

                            padded_grad[
                                : full_grad.numel()
                            ] = full_grad

                            full_grad = padded_grad

                        # backward 已经用完完整 weight，
                        # 现在可以重新只保存 local master shard
                        self._restore_local_shard(
                            current_info
                        )

                        # 暂时不要让 optimizer 看到完整 grad
                        param.grad = None

                        # -------------------------------------------------
                        # Gloo fallback
                        # -------------------------------------------------
                        #
                        # tests 使用 backend="gloo"。
                        #
                        # 数学上：
                        #
                        # all_reduce(full_grad)
                        # + 只切出本 rank 对应的一段
                        #
                        # 与 reduce-scatter 得到的结果完全相同。
                        #
                        # 在 NCCL / 真正 GPU FSDP 中，
                        # 我们更希望直接 reduce_scatter，
                        # 因为这样通信/内存更有效。
                        # -------------------------------------------------

                        handle = dist.all_reduce(
                            full_grad,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )

                        self._pending_grad_sync.append(
                            {
                                "handle": handle,
                                "param": param,
                                "full_grad": full_grad,
                                "info": current_info,
                            }
                        )

                    return grad_hook

                param.register_post_accumulate_grad_hook(
                    make_grad_hook(info)
                )

    # ============================================================
    # 5. 普通 forward
    # ============================================================

    def forward(
        self,
        *inputs,
        **kwargs,
    ):
        # 真正的 gather / reshard 都由 hooks 做
        return self.module(
            *inputs,
            **kwargs,
        )

    # ============================================================
    # 6. backward 后完成梯度同步
    # ============================================================

    def finish_gradient_synchronization(self):

        # --------------------------------
        # A. 完成 sharded parameter grad
        # --------------------------------

        for item in self._pending_grad_sync:

            handle = item["handle"]
            param = item["param"]
            full_grad = item["full_grad"]
            info = item["info"]

            # 等异步 all_reduce 完成
            handle.wait()

            shard_size = info["shard_size"]

            start = (
                self.rank
                * shard_size
            )

            end = start + shard_size

            # 只保留当前 rank 对应的 gradient shard
            local_grad = (
                full_grad[start:end]
                .clone()
            )

            # SUM -> average
            local_grad /= self.world_size

            # master weight 是 fp32，
            # optimizer gradient 也保持 fp32
            local_grad = local_grad.to(
                param.data.dtype
            )

            param.grad = local_grad

        self._pending_grad_sync.clear()

        # --------------------------------
        # B. 未 shard 的参数
        # --------------------------------
        #
        # 比如 RMSNorm.weight。
        #
        # 它们仍然在每个 rank 上完整复制，
        # 所以要走普通 DDP all-reduce。
        # --------------------------------

        sharded_param_ids = {
            id(info["param"])
            for info in self._sharded_params.values()
        }

        for param in self.module.parameters():

            if not param.requires_grad:
                continue

            if id(param) in sharded_param_ids:
                continue

            if param.grad is None:
                continue

            dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
            )

            param.grad /= self.world_size

    # ============================================================
    # 7. 为测试恢复完整 state_dict
    # ============================================================

    def gather_full_params(
        self,
    ) -> dict[str, torch.Tensor]:

        result = {}

        sharded_param_ids = {
            id(info["param"]): info
            for info in self._sharded_params.values()
        }

        for name, param in self.module.named_parameters():

            # -------------------------
            # 普通 replicated 参数
            # -------------------------
            if id(param) not in sharded_param_ids:
                result[name] = (
                    param.data
                    .detach()
                    .clone()
                )

                continue

            # -------------------------
            # sharded parameter
            # -------------------------

            info = sharded_param_ids[
                id(param)
            ]

            local = (
                param.data
                .detach()
            )

            gathered = [
                torch.empty_like(local)
                for _ in range(
                    self.world_size
                )
            ]

            dist.all_gather(
                gathered,
                local,
            )

            full_flat = torch.cat(
                gathered,
                dim=0,
            )

            full_flat = full_flat[
                : info["original_numel"]
            ]

            full = full_flat.reshape(
                info["original_shape"]
            )

            result[name] = full.clone()

        return result