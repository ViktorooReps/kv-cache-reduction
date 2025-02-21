# coding=utf-8
# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, TypedDict

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from transformers.utils import logging


logger = logging.get_logger(__name__)


def _noop(score, b, h, q_idx, kv_idx):
    return score


def _get_kv_bias_mod(kv_bias: torch.Tensor):
    def kv_bias_mod(score, b, h, q_idx, kv_idx):
        return score + kv_bias[b, h, kv_idx]
    return kv_bias_mod


def flex_attention_with_kv_bias(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: BlockMask = None,
    kv_bias: Optional[torch.Tensor] = None,  # (B, H, KV)
    **kwargs,
):
    _, n_q_heads, _, _ = query_states.shape
    _, n_kv_heads, _, _ = key_states.shape

    enable_gqa = (n_q_heads != n_kv_heads)

    score_mod = _noop
    if kv_bias is not None:
        score_mod = _get_kv_bias_mod(kv_bias)

    attn_output = flex_attention(
        query_states, key_states, value_states, score_mod=score_mod, block_mask=attention_mask, enable_gqa=enable_gqa
    )

    return attn_output.transpose(1, 2).contiguous(), None


class FlexAttentionKwargs(TypedDict, total=False):
    kv_bias: Optional[torch.Tensor]
