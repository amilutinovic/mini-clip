import os
import json

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

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
        
from transformers import AutoModel
from .image_encoder import build_projection_head  # isti pattern kao za image encoder

class TextEncoderPretrained(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", emb_dim=256, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 768 za distilbert-base
        self.projection = build_projection_head(hidden_size, emb_dim)

    def forward(self, input_ids, attention_mask):
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # masked mean pooling preko tokena (isti princip kao u MiniTextTransformer)
        token_embeddings = out.last_hidden_state          # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        x = self.projection(pooled)
        return F.normalize(x, dim=-1)
        
class ContrastiveFlickrDatasetBERT(Dataset):
    def __init__(self, df, images_dir, tokenizer, max_length=32, transform=None, random_caption=True):
        self.df = df
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.random_caption = random_caption
        self.image_names = self.df["image"].unique()
        self.captions_by_image = {
            name: group.reset_index(drop=True)
            for name, group in self.df.groupby("image")
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        captions = self.captions_by_image[image_name]
        row_index = random.randrange(len(captions)) if self.random_caption else 0
        row = captions.iloc[row_index]

        image = Image.open(os.path.join(self.images_dir, image_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        encoded = self.tokenizer(
            row["caption"], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "caption": row["caption"],
            "image_name": image_name
        }
        
        
