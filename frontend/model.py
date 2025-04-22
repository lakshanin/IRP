#!/usr/bin/env python3
"""
Defines the MultiTaskConformer model for joint phoneme recognition and error classification.
Components:
  • Conformer-based encoder with stacked blocks for feature extraction.
  • CTC head for phoneme sequence prediction.
  • Cross-attention error classification head over canonical phoneme embeddings.
"""

import math
import torch
import torch.nn as nn


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Feed Forward Module
class FeedForwardModule(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# Convolution Module
class ConvolutionModule(nn.Module):
    def __init__(self, dim_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_model)
        self.pointwise_conv1 = nn.Conv1d(dim_model, 2 * dim_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            dim_model, dim_model,
            kernel_size=kernel_size,
            groups=dim_model,
            padding=kernel_size // 2
        )
        self.batch_norm = nn.GroupNorm(1, dim_model)
        self.pointwise_conv2 = nn.Conv1d(dim_model, dim_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return self.dropout(x.transpose(1, 2))


# Multi-Headed Self-Attention Module
class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, dim_model, num_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_model)
        self.self_attn = nn.MultiheadAttention(
            dim_model, num_heads,
            dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_norm = self.layer_norm(x)
        out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        return x + self.dropout(out)


# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(self, dim_model, dim_ff, num_heads,
                 kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim_model, dim_ff, dropout)
        self.ffn2 = FeedForwardModule(dim_model, dim_ff, dropout)
        self.attn = MultiHeadedSelfAttentionModule(dim_model, num_heads, dropout)
        self.conv = ConvolutionModule(dim_model, kernel_size, dropout)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
        self.scaling_factor = 0.5

    def forward(self, x, mask=None):
        x = x + self.scaling_factor * self.dropout(self.ffn1(x))
        x = x + self.attn(x, mask=mask)
        x = x + self.conv(x)
        x = x + self.scaling_factor * self.dropout(self.ffn2(x))
        return self.norm(x)


# Conformer Encoder
class ConformerEncoder(nn.Module):
    def __init__(self, num_blocks=6, dim_model=512, dim_ff=2048,
                 num_heads=8, kernel_size=15, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConformerBlock(dim_model, dim_ff, num_heads, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return x


# Multi-Task Conformer
class MultiTaskConformer(nn.Module):
    def __init__(self,
                 input_dim=80,
                 num_blocks=6,
                 dim_model=512,
                 dim_ff=2048,
                 num_heads=8,
                 kernel_size=15,
                 dropout=0.1,
                 num_ctc_classes=40,
                 num_error_classes=3):
        super().__init__()
        self.num_ctc_classes = num_ctc_classes
        self.num_error_classes = num_error_classes

        # Encoder
        self.linear_in = nn.Linear(input_dim, dim_model)
        self.pos_encoding = PositionalEncoding(dim_model, dropout=dropout)
        self.encoder = ConformerEncoder(
            num_blocks=num_blocks,
            dim_model=dim_model,
            dim_ff=dim_ff,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.ctc_out = nn.Linear(dim_model, num_ctc_classes)

        # Error classification head
        self.canonical_embedding = nn.Embedding(num_ctc_classes, dim_model)
        self.cross_attn = nn.MultiheadAttention(dim_model, num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.error_out = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, num_error_classes)
        )

    def forward(self, feats, feat_lengths, canonical_ids=None, canonical_lengths=None):
        x = self.linear_in(feats)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        ctc_logits = self.ctc_out(x)

        error_logits = None
        if canonical_ids is not None:
            canon_emb = self.canonical_embedding(canonical_ids)
            attn_output, _ = self.cross_attn(query=canon_emb, key=x, value=x)
            error_logits = self.error_out(attn_output)

        return {
            "ctc_logits": ctc_logits,
            "error_logits": error_logits
        }
