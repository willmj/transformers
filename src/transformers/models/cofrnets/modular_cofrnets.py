from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaAttention,
)

from .cofrnet_modules.CoFrNet_continuants import CoFrNetContinuant


class CoFrNetsConfig(LlamaConfig):
    model_type = "cofrnets"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        # CoFrNet flags
        use_cofr_mlp: bool = True,
        use_cofr_attention: bool = True,
        cofrnet_dim: Optional[int] = None,
        cofr_mlp_width: int = 1,
        cofr_mlp_depth: int = 1,
        cofr_mlp_epsilon: float = 0.1,
        cofr_mlp_variant: str = "fully_connected",
        cofr_attention_width: int = 1,
        cofr_attention_depth: int = 1,
        cofr_attention_epsilon: float = 0.1,
        cofr_attention_variant: str = "fully_connected",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )

        self.use_cofr_mlp = use_cofr_mlp
        self.use_cofr_attention = use_cofr_attention

        self.cofr_mlp_width = cofr_mlp_width
        self.cofr_mlp_depth = cofr_mlp_depth
        self.cofr_mlp_epsilon = cofr_mlp_epsilon
        self.cofr_mlp_variant = cofr_mlp_variant

        self.cofr_attention_width = cofr_attention_width
        self.cofr_attention_depth = cofr_attention_depth
        self.cofr_attention_epsilon = cofr_attention_epsilon
        self.cofr_attention_variant = cofr_attention_variant

        self.cofrnet_dim = (
            cofrnet_dim
            if cofrnet_dim is not None else max_position_embeddings
        )


class CoFrNetsMLP(nn.Module):
    def __init__(self, config: CoFrNetsConfig):
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        use_bias = bool(getattr(config, "mlp_bias", False))

        self.w1 = nn.Linear(hidden, hidden, bias=use_bias)
        self.wg = nn.Linear(hidden, hidden, bias=use_bias)

        self.act = nn.SiLU()

        self.input_dim = hidden
        self.cofrnet = CoFrNetContinuant(
            input_dim=hidden,
            output_dim=hidden,
            width=getattr(config, "cofr_mlp_width", 1),
            depth=getattr(config, "cofr_mlp_depth", 1),
            epsilon=getattr(config, "cofr_mlp_epsilon", 0.1),
            variant=getattr(config, "cofr_mlp_variant", None),
        )

        for layer in [self.w1, self.wg]:
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, hidden_states):
        mlp_output = self.cofrnet(self.w1(hidden_states) * self.act(self.wg(hidden_states)), self.input_dim)
        return mlp_output

class CoFrNetsAttention(nn.Module):
    def __init__(self, config: CoFrNetsConfig, layer_idx: Optional[int] = None):
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.out_grid = int(config.cofrnet_dim)

        self.input_dim = hidden
        self.cofrnet = CoFrNetContinuant(
            input_dim=hidden,
            output_dim=self.out_grid,
            width=config.cofr_attention_width,
            depth=config.cofr_attention_depth,
            epsilon=config.cofr_attention_epsilon,
            variant=config.cofr_attention_variant,
        )
        self.value_proj = nn.Linear(hidden, hidden, bias=True)

        nn.init.trunc_normal_(self.value_proj.weight, mean=0.0, std=0.02)
        if self.value_proj.bias is not None:
            self.value_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        cofr_scores = self.cofrnet(hidden_states, self.input_dim)
        upper_mask = torch.triu(torch.ones(self.out_grid, self.out_grid), diagonal=1).bool().to(cofr_scores.device)
        upper_mask = upper_mask.unsqueeze(0)
        cofr_scores = cofr_scores.masked_fill(upper_mask[:,:cofr_scores.size(dim=1),:], float('-inf'))
        cofr_scores = F.softmax(cofr_scores, dim=-1)
        value_states = self.value_proj(hidden_states)
        attn_output = torch.bmm(cofr_scores[:, :cofr_scores.size(dim=1), :cofr_scores.size(dim=1)], value_states)
        return attn_output, None


class CoFrNetsDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: CoFrNetsConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = CoFrNetsMLP(config)
        self.self_attn = CoFrNetsAttention(config=config, layer_idx=layer_idx)


class CoFrNetsModel(LlamaModel):
    def __init__(self, config: CoFrNetsConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CoFrNetsDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


class CoFrNetsForCausalLM(LlamaForCausalLM):
    def __init__(self, config: CoFrNetsConfig):
        super().__init__(config)
        self.model = CoFrNetsModel(config)
