import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_length):
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        

class MiniTextTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_length,
        padding_idx,
        embedding_dim=256,
        num_heads=4,
        num_layers=2,
        feedforward_dim=512,
        output_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.position_encoding = PositionalEncoding(
            d_model=embedding_dim,
            max_length=max_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.projection = nn.Linear(
            embedding_dim,
            output_dim
        )

    def forward(self, input_ids, attention_mask):

        # [B, L]
        x = self.embedding(input_ids)

        # [B, L, D]
        x = self.position_encoding(x)

        padding_mask = attention_mask == 0

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # [B, L, D] -> [B, D]
        mask = attention_mask.unsqueeze(-1).float()

        x = x * mask

        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # [B, D] -> [B, output_dim]
        x = self.projection(x)

        x = F.normalize(x, dim=-1)

        return x
