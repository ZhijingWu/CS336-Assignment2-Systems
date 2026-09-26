# scripts/sharded_optimizer.py

from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):

    def __init__(
        self,
        params,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # 所有参数
        self.all_params = []

        # 当前 rank 负责更新的参数
        self.local_params = []

        # param -> owner rank
        self.param_to_rank = {}

        # 用于给参数做全局编号
        self._param_index = 0

        # 父类会调用 add_param_group
        super().__init__(params, defaults={})

        # 这里只把当前 rank 负责的参数交给真正的 optimizer
        self.inner_optimizer = optimizer_cls(
            self.local_params,
            **kwargs,
        )

    def add_param_group(
        self,
        param_group: dict[str, Any],
    ):
        params = param_group["params"]

        # 防止 generator 只能遍历一次
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        # 给每个参数指定 owner rank
        for param in params:
            owner_rank = (
                self._param_index
                % self.world_size
            )

            self.all_params.append(param)
            self.param_to_rank[param] = owner_rank

            if owner_rank == self.rank:
                self.local_params.append(param)

            self._param_index += 1

        # Optimizer 父类自己也要记录完整 param group
        new_group = dict(param_group)
        new_group["params"] = params

        super().add_param_group(new_group)

    @torch.no_grad()
    def step(
        self,
        closure=None,
        **kwargs,
    ):
        # 当前 rank 只更新自己负责的参数
        loss = self.inner_optimizer.step(
            closure=closure,
            **kwargs,
        )

        # 每个参数由它的 owner rank
        # 把更新后的值广播给所有其他 rank
        for param in self.all_params:
            owner_rank = self.param_to_rank[param]

            dist.broadcast(
                param.data,
                src=owner_rank,
            )

        return loss

    def zero_grad(
        self,
        set_to_none: bool = True,
    ):
        # 注意模型的所有参数都可能有 grad，
        # 所以这里清整个 wrapper 里的参数梯度
        super().zero_grad(
            set_to_none=set_to_none
        )