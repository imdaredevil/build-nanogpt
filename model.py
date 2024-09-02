# implement gpt 2 model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class CausualSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn.RESIDUAL = True
        self.c_proj.RESIDUAL = True

    def forward(self, x):
        """
        Forward pass of the CausualSelfAttention module.
        """
        q, k, v = self.c_attn(x).chunk(3, dim = -1)
        q = q.view(-1, x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(-1, x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(-1, x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # mask out future tokens
        scores = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        # scores = scores.masked_fill(mask == 0, -np.inf)
        # scores = F.softmax(scores, dim = -1)
        # scores = torch.matmul(scores, v)
        scores = scores.transpose(1, 2).contiguous().view(-1, x.shape[1], self.n_embd)
        return self.c_proj(scores)


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausualSelfAttention(config)
        self.mlp = nn.ModuleDict({
            "c_fc": nn.Linear(config.n_embd, config.n_inner),
            "c_proj": nn.Linear(config.n_inner, config.n_embd),
        })
        self.mlp["c_proj"].RESIDUAL = True
        self.mlp["c_fc"].RESIDUAL = True
        self.mlp_gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        """
        Forward pass of the Block module.

        Parameters
        ----------
        x : torch.tensor
            Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns
        -------
        x : torch.tensor
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = x + self.attn(self.ln_1(x))
        mlp = self.mlp_gelu(self.mlp["c_fc"](self.ln_2(x)))
        x = x + self.mlp["c_proj"](mlp)
        return x


class GPT2Transformer(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_layer = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, input_ids):
        x = self.wte(input_ids)
        x = x + self.wpe(torch.arange(x.shape[1], device=x.device))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        if config.n_inner is None:
            config.n_inner = 4 * config.n_embd
        self.transformer = GPT2Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # Fun note here: When we share weights, use the linear layer weights to initialize the weight. If you do it other way like below, the starting point of the optimization will be different.
        # Specifically, the linear model uses Xavier initialization. But, the embedding layer uses normal initialization. Normal initialization is not favourable for linear layers. But, Xavier initialization work for embedding layers.
        # Thats why we should not use the below line
        # self.lm_head.weight = self.transformer.wte.weight
        self.config = config
        for module in self.modules():
            self._init_weights(module)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL"):
                std *= (self.config.n_layer ** -0.5)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids):
        # pad sequences to max sequence length
        x = self.transformer(input_ids)
        return self.lm_head(x)
