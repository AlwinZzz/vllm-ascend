# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ulysses-style local tensor-parallel exchange for the DSV4 wq_b projection.

Under attention DP (``tensor_parallel_size == 1``) with
``additional_config.finegrained_tp_config.qb_tensor_parallel_size = k``, every
rank keeps its own tokens but only holds a contiguous 1/k shard of the wq_b
output heads. The QBMM then runs as:

    gather tokens : AllGather qr / cos / sin across the QB group, padded to a
                    static exchange capacity so ACL graph shapes stay fixed
    local matmul  : wq_b shard over all gathered tokens (weight reads drop to
                    1/k while FLOPs per rank are unchanged)
    return heads  : all-to-all routes each head shard back to the token owner

Only token-major data (qr and its per-token rope cos/sin) needs gathering;
head-shared parameters (q_b_norm weight) stay local. Zero-padded rows are
numerically inert because wq_b has no bias.
"""

import torch
import torch.distributed as dist

from vllm_ascend.distributed.parallel_state import get_qb_tp_group
from vllm_ascend.utils import get_potential_max_tokens


class QBTPExchange:
    """Static-buffer token gather / head return exchange for the QB TP group.

    Buffers are lazily allocated on first use (the profiling run always
    happens before ACL graph capture) and keep stable device addresses across
    capture/replay cycles, mirroring the o_proj fine-grained TP path
    (``AscendDSAImpl._forward_o_proj``). One instance is shared by all layers:
    layers execute sequentially, so replays never overlap on a buffer.
    """

    def __init__(self, qb_tp_size: int):
        self.qb_tp_size = qb_tp_size
        self._buffers: dict[tuple, torch.Tensor] = {}

    def _static_buf(self, key: tuple, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.zeros(shape, dtype=dtype, device=device)
            self._buffers[key] = buf
        return buf

    def _all_gather_tokens(self, tensor: torch.Tensor, name: str, exchange_num_tokens: int) -> torch.Tensor:
        num_tokens = tensor.shape[0]
        tail = tuple(tensor.shape[1:])
        send = self._static_buf(
            ("qb_send", name, tail, tensor.dtype), (exchange_num_tokens, *tail), tensor.dtype, tensor.device
        )
        send.zero_()
        send[:num_tokens].copy_(tensor)
        gathered = self._static_buf(
            ("qb_recv", name, tail, tensor.dtype),
            (self.qb_tp_size * exchange_num_tokens, *tail),
            tensor.dtype,
            tensor.device,
        )
        dist.all_gather_into_tensor(gathered, send, group=get_qb_tp_group().device_group)
        return gathered

    def gather_tokens(
        self, qr: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad + AllGather token-major inputs; returns gathered-row tensors."""
        exchange_num_tokens = get_potential_max_tokens()
        num_tokens = qr.shape[0]
        if num_tokens > exchange_num_tokens:
            raise ValueError(
                "qb static exchange capacity must cover local tokens, "
                f"got {exchange_num_tokens} and {num_tokens}."
            )
        return (
            self._all_gather_tokens(qr, "qr", exchange_num_tokens),
            self._all_gather_tokens(cos, "cos", exchange_num_tokens),
            self._all_gather_tokens(sin, "sin", exchange_num_tokens),
        )

    def return_heads(self, q: torch.Tensor, num_tokens: int, n_local_heads: int) -> torch.Tensor:
        """All-to-all head shards back to their token owners.

        Args:
            q: [qb_tp_size * E, n_local_heads, head_dim], gathered-token-major.
            num_tokens: local (owner) token count, used to trim the padding.
            n_local_heads: head count of one shard (n_heads // qb_tp_size).

        Returns:
            [num_tokens, n_heads, head_dim] with heads in their original order
            (shard r holds the contiguous heads [r * n_local_heads, ...)).
        """
        exchange_num_tokens = get_potential_max_tokens()
        head_dim = q.shape[-1]
        shape = (self.qb_tp_size, exchange_num_tokens, n_local_heads, head_dim)
        send = self._static_buf(("qb_a2a_send", q.dtype), shape, q.dtype, q.device)
        send.copy_(q.reshape(shape))
        recv = self._static_buf(("qb_a2a_recv", q.dtype), shape, q.dtype, q.device)
        dist.all_to_all_single(recv.view(-1), send.view(-1), group=get_qb_tp_group().device_group)
        return (
            recv[:, :num_tokens]
            .transpose(0, 1)
            .reshape(num_tokens, self.qb_tp_size * n_local_heads, head_dim)
            .contiguous()
        )


_QB_TP_EXCHANGE: QBTPExchange | None = None


def get_qb_tp_exchange(qb_tp_size: int) -> QBTPExchange:
    """Process-wide singleton: all layers share one set of static buffers."""
    global _QB_TP_EXCHANGE
    if _QB_TP_EXCHANGE is None:
        _QB_TP_EXCHANGE = QBTPExchange(qb_tp_size)
    return _QB_TP_EXCHANGE
