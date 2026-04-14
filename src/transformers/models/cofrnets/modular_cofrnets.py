# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

import torch
from torch import nn

from ...activations import ACT2FN
from ..mixtral.configuration_mixtral import MixtralConfig
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralSparseMoeBlock,
)
from .cofrnet_modules.CoFrNet_continuants import CoFrNetContinuant


class CoFrNetsConfig(MixtralConfig):
    model_type = "cofrnets"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 3072,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        sliding_window: Optional[int] = None,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 1,
        num_local_experts: int = 4,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        rope_parameters: Optional[dict] = None,
        cofr_mlp_width: int = 1,
        cofr_mlp_depth: int = 1,
        cofr_mlp_epsilon: float = 0.1,
        cofr_mlp_variant: str = "fully_connected",
        **kwargs,
    ):
        if intermediate_size is None:
            intermediate_size = hidden_size

        kwargs.setdefault("rope_theta", 500000.0)

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_local_experts,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            router_jitter_noise=router_jitter_noise,
            rope_parameters=rope_parameters,
            **kwargs,
        )

        self.cofr_mlp_width = cofr_mlp_width
        self.cofr_mlp_depth = cofr_mlp_depth
        self.cofr_mlp_epsilon = cofr_mlp_epsilon
        self.cofr_mlp_variant = cofr_mlp_variant


class CoFrNetsMLP(nn.Module):
    """Per-expert MLP: fused gate/up + CoFrNet-replaced down projection."""

    def __init__(self, config: CoFrNetsConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.cofrnet = CoFrNetContinuant(
            input_dim=self.ffn_dim,
            output_dim=self.hidden_dim,
            width=config.cofr_mlp_width,
            depth=config.cofr_mlp_depth,
            epsilon=config.cofr_mlp_epsilon,
            variant=config.cofr_mlp_variant,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        # CoFrNetContinuant expects (batch, seq, input_dim); experts pass 2D (tokens, dim)
        squeeze_batch = intermediate.dim() == 2
        if squeeze_batch:
            intermediate = intermediate.unsqueeze(0)
        output = self.cofrnet(intermediate, self.ffn_dim)
        if squeeze_batch:
            output = output.squeeze(0)
        return output


class CoFrNetsExperts(MixtralExperts):
    pass


class CoFrNetsSparseMoeBlock(MixtralSparseMoeBlock):
    pass


class CoFrNetsDecoderLayer(MixtralDecoderLayer):
    pass


class CoFrNetsPreTrainedModel(MixtralPreTrainedModel):
    pass


class CoFrNetsModel(MixtralModel):
    pass


class CoFrNetsForCausalLM(MixtralForCausalLM):
    pass


__all__ = [
    "CoFrNetsConfig",
    "CoFrNetsPreTrainedModel",
    "CoFrNetsModel",
    "CoFrNetsForCausalLM",
]
