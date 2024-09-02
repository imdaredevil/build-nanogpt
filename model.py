# implement gpt 2 model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class CausualSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd) 

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
        mask = torch.tril(torch.ones((x.shape[1], x.shape[1]))).view(1, 1, x.shape[1], x.shape[1]).to(self.c_proj.weight.device)
        scores = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # scores = scores.masked_fill(mask == 0, -np.inf)
        # scores = F.softmax(scores, dim = -1)
        # scores = torch.matmul(scores, v)
        scores = scores.transpose(1, 2).contiguous().view(-1, x.shape[1], self.n_embd)
        return self.c_proj(scores)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, hidden_dim = 3072):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.attn = CausualSelfAttention(n_embd, n_head)
        self.mlp = nn.ModuleDict({
            "c_fc": nn.Linear(n_embd, hidden_dim),
            "c_proj": nn.Linear(hidden_dim, n_embd),
        })
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
    def __init__(self, n_layer = 12, vocab_size = 50257, n_head = 12, n_embd = 768, n_ctx = 1024, n_positions = 1024, **kwargs):
        super().__init__()
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_ctx = n_ctx
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList([Block(n_embd, n_head, hidden_dim = 4 * n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

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
        self.transformer = GPT2Transformer(**config.to_dict())
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids):
        # pad sequences to max sequence length
        x = self.transformer(input_ids)
        return self.lm_head(x)
