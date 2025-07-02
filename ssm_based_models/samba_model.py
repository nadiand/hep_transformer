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

# from einops import rearrange
# from causal_conv1d import causal_conv1d_fn

FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

class Model(nn.Module):
    def __init__(self, input_size, d_model, n_layers, layer_norm_eps, n_head, full_per_layer, local_window, output_size):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.block1 = Block(d_model, n_layers, layer_norm_eps, 1, full_per_layer, local_window, n_head)
        self.block2 = Block(d_model, n_layers, layer_norm_eps, 2, full_per_layer, local_window, n_head)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block1(x)
        x = self.block2(x)
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
        self.attn = CausalSelfAttention(d_model, local_window, full_per_layer, n_head, layer_idx)
        self.mlp = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        out = self.mb(self.norm_m(x.to(dtype=self.norm_m.weight.dtype))) + x.to(torch.float32)
        out = out + self.attn(self.norm_attn(out.to(dtype=self.norm_attn.weight.dtype)), mask)
        out = out + self.mlp(self.norm_mlp(out.to(dtype=self.norm_mlp.weight.dtype)))
        return out



class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, local_window, full_per_layer, n_head, layer_idx, attn_bias = True):
        super().__init__()
        self.local = layer_idx % full_per_layer < full_per_layer-1
        self.local_window = local_window
        self.head_size = n_embd // n_head
        self.n_head =  n_head
        self.n_query_groups = self.n_head
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=attn_bias)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd, bias=attn_bias)
        self.sc = False # NOTE This is the default value in the samba config files, so I'll comment out everything that needs self.sc
#        if self.sc:
#            self.q_dim = self.n_head * self.head_size
#            self.kv_dim = self.n_query_groups * self.head_size
#            d_conv = 4
#            self.q_conv1d = nn.Conv1d(
#                in_channels=self.q_dim,
#                out_channels=self.q_dim,
#                bias=False,
#                kernel_size=d_conv,
#                groups=self.q_dim,
#                padding=d_conv - 1,
#            )
#            self.k_conv1d = nn.Conv1d(
#                in_channels=self.kv_dim,
#                out_channels=self.kv_dim,
#                bias=False,
#                kernel_size=d_conv,
#                groups=self.kv_dim,
#                padding=d_conv - 1,
#            )
#            self.v_conv1d = nn.Conv1d(
#                in_channels= self.kv_dim,
#                out_channels= self.kv_dim,
#                bias=False,
#                kernel_size=d_conv,
#                groups= self.kv_dim,
#                padding=d_conv - 1,
#            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1 )
        v = v.reshape(B,  T, -1 )
        # if self.sc:
        #     q = causal_conv1d_fn(
        #                 x = q.transpose(-1,-2),
        #                 weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
        #                 bias=self.q_conv1d.bias,
        #                 activation="silu",
        #             ).transpose(-1,-2)
        #     k = causal_conv1d_fn(
        #                 x = k.transpose(-1,-2),
        #                 weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
        #                 bias=self.k_conv1d.bias,
        #                 activation="silu",
        #             ).transpose(-1,-2)
        #     v = causal_conv1d_fn(
        #                 x = v.transpose(-1,-2),
        #                 weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
        #                 bias=self.v_conv1d.bias,
        #                 activation="silu",
        #             ).transpose(-1,-2)

        q = q.reshape(B,  T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.head_size)
        v = v.reshape(B,  T, -1, self.head_size)

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)

        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            print("flash!")
            from flash_attn import flash_attn_func
            if self.local and self.local_window > -1:
                win_tuple = (self.local_window-1, 0)
            else:
                win_tuple = (-1,-1)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True, window_size=win_tuple)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


model = Model(input_size=3, d_model=128, n_layers=3, layer_norm_eps=0.1, n_head=4, full_per_layer=2, local_window=10, output_size=3)
print("total params", sum(p.numel() for p in model.parameters() if p.requires_grad))