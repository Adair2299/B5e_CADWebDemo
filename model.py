"""A small causal Transformer for question-conditioned English generation."""

import torch
from torch import nn


class TinyChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_length=48, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                           n_layers=n_layers, max_length=max_length, dropout=dropout)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch_size, length = token_ids.shape
        if length > self.config["max_length"]:
            raise ValueError("Input exceeds max_length.")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(length, length, device=token_ids.device,
                                           dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=causal_mask)
        return self.output(self.norm(x))
