from typing import Optional, Union, Callable
import torch
from torch.nn import Module, Linear, LayerNorm, Dropout, TransformerEncoder
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention


def compile_flex_attention():
    try:
        return torch.compile(flex_attention)
    except:
        return torch.compile(flex_attention, dynamic=False, mode="max-autotune")


class TransformerEncoderLayer(Module):
    """
    Code taken and adapted from official pytorch implementation of TransformerEncoderLayer:
    https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = CausalSelfAttention(nhead, d_model, bias=True, batch_first=batch_first, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            flex_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), flex_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, flex_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, flex_mask: Optional[Tensor],
                  attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, flex_mask)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class CausalSelfAttention(Module):
    """
    Taken and adapted from pytorch tutorial on SDPA:
    https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#beta-implementing-high-performance-transformers-with-scaled-dot-product-attention-sdpa
    """

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, batch_first: bool=True, dropout: float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal
        self.batch_first = batch_first
        self.dropout = dropout
        self.resid_dropout = Dropout(dropout)

    def forward(self, x, flex_mask):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # Using flex attention (non causal currently!)
        compiled_flex_attention = compile_flex_attention()
        y = compiled_flex_attention(query, key, value, block_mask=flex_mask)

        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)
        y = self.resid_dropout(self.c_proj(y))

        return y


def generate_flex_padding_mask(seqlens):
    def padding_mask(b, h, q_idx, kv_idx):
        length = seqlens[b]
        padding = (kv_idx < length) & (q_idx < length)
        return padding
    return padding_mask


def generate_flex_window_padding_mask(seqlens, window_size):
    def sliding_window_padded(b, h, q_idx, kv_idx):
        length = seqlens[b]
        padding = (kv_idx < length) & (q_idx < length)
        window_mask = torch.abs(q_idx - kv_idx) <= window_size
        return padding & window_mask
    return sliding_window_padded


class CustomTransformerEncoder(TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers.

    .. note::
        See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
        for an in depth discussion of the performant building blocks PyTorch offers for building your own
        transformer layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm, enable_nested_tensor=enable_nested_tensor)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        flex_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
                flex_mask=flex_mask,
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output
