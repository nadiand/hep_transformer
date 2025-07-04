# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from mambapy.mamba import Mamba, MambaConfig

FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

class Model(nn.Module):
    def __init__(self, input_size, d_model, n_layers, layer_norm_eps, n_head, full_per_layer, local_window, output_size):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.block1 = Block(d_model, n_layers, layer_norm_eps, 1, full_per_layer, local_window, n_head)
        self.block2 = Block(d_model, n_layers, layer_norm_eps, 2, full_per_layer, local_window, n_head)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, x, mask):
        x = self.input_layer(x)
        x = self.block1(x, mask)
        x = self.block2(x, mask)
        out = self.decoder(x)
        return out

class Block(nn.Module):
    def __init__(self, d_model, n_layers, layer_norm_eps, layer_idx, full_per_layer, local_window, n_head):
        # NOTE: leaving only the mamba swa mlp option ! not fully samba
        super().__init__()
        factory_kwargs = {"device": "cuda", "dtype": torch.float32}

        self.norm_m = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm_attn = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm_mlp = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.mb = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.attn = CausalSelfAttention(n_head, d_model, batch_first=False, dropout=0.1)
        self.mlp = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        out = self.mb(self.norm_m(x.to(dtype=self.norm_m.weight.dtype))) + x
        out = out + self.attn(self.norm_attn(out.to(dtype=self.norm_attn.weight.dtype)))
        out = out + self.mlp(self.norm_mlp(out.to(dtype=self.norm_mlp.weight.dtype)))
        return out


class CausalSelfAttention(nn.Module):
    '''
    Taken and adapted from pytorch tutorial on SDPA:
    https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#beta-implementing-high-performance-transformers-with-scaled-dot-product-attention-sdpa
    '''

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, batch_first: bool=True, dropout: float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal
        self.batch_first = batch_first
        self.dropout = dropout

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        # The logic ensuring flash attention is utilized
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)

        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        return y