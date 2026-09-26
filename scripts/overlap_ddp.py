import torch
import torch.nn as nn
import torch.distributed as dist


class OverlapDDP(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        for param in self.module.parameters():
            dist.broadcast(
                param.data,
                src=0,
            )

        # 每个参数的梯度 ready 后自动触发 hook
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._grad_hook
                )

    def forward(self, *args, **kwargs):
        return self.module(
            *args,
            **kwargs,
        )

    def _grad_hook(self, param):

        if param.grad is None:
            return

        handle = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        self.handles.append(
            (handle, param)
        )

    def finish_gradient_synchronization(self):

        for handle, param in self.handles:
            handle.wait()

            param.grad /= self.world_size

        self.handles.clear()