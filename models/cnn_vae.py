"""
cnn_vae.py — A1 CNN encoder/decoder.

Architecture:
  Encoder: 3× ConvBlock(stride=2) → shared features at (B, 128, 8, 8)
           → 3× ConvHead: common(4ch), disease1(2ch), disease2(2ch)
  Decoder: concat z_c + z_d1 + z_d2 → 3× ConvTransposeBlock → Tanh

Input/output: (B, 1, 64, 64) for SynDis/SpyMNIST; set in_ch=3 for RGB tiers.
Latent spatial size: 8×8 (after 3× stride-2 with 64×64 input).
"""
import torch
import torch.nn as nn

from models.shared import ConvBlock, ConvTransposeBlock, ConvHead, LabelAttentionHead
from losses.base import reparameterize


class CNNEncoder(nn.Module):

    def __init__(
        self,
        in_ch:      int = 1,
        base_ch:    int = 32,
        z_common:   int = 4,
        z_disease:  int = 2,
        use_label_attn: bool = False,
        attn_query_dim: int  = 64,
    ):
        super().__init__()
        self.use_label_attn = use_label_attn

        self.backbone = nn.Sequential(
            ConvBlock(in_ch,       base_ch,     stride=2),   # 64→32
            ConvBlock(base_ch,     base_ch * 2, stride=2),   # 32→16
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),   # 16→8
        )
        feat_ch = base_ch * 4  # 128 by default

        self.head_common   = ConvHead(feat_ch, z_common)
        if use_label_attn:
            self.head_d1 = LabelAttentionHead(feat_ch, z_disease, attn_query_dim)
            self.head_d2 = LabelAttentionHead(feat_ch, z_disease, attn_query_dim)
        else:
            self.head_d1 = ConvHead(feat_ch, z_disease)
            self.head_d2 = ConvHead(feat_ch, z_disease)

    def forward(self, x: torch.Tensor, label_d1=None, label_d2=None):
        """
        Returns dict with keys: mu_c, lv_c, mu_d1, lv_d1, mu_d2, lv_d2,
        and optionally attn_d1, attn_d2 (if use_label_attn).
        """
        feat = self.backbone(x)
        mu_c, lv_c = self.head_common(feat)

        if self.use_label_attn and label_d1 is not None:
            mu_d1, lv_d1, attn_d1 = self.head_d1(feat, label_d1)
            mu_d2, lv_d2, attn_d2 = self.head_d2(feat, label_d2)
            return dict(mu_c=mu_c, lv_c=lv_c,
                        mu_d1=mu_d1, lv_d1=lv_d1,
                        mu_d2=mu_d2, lv_d2=lv_d2,
                        attn_d1=attn_d1, attn_d2=attn_d2)
        else:
            mu_d1, lv_d1 = self.head_d1(feat)
            mu_d2, lv_d2 = self.head_d2(feat)
            return dict(mu_c=mu_c, lv_c=lv_c,
                        mu_d1=mu_d1, lv_d1=lv_d1,
                        mu_d2=mu_d2, lv_d2=lv_d2)


class CNNDecoder(nn.Module):

    def __init__(
        self,
        out_ch:    int = 1,
        base_ch:   int = 32,
        z_common:  int = 4,
        z_disease: int = 2,
    ):
        super().__init__()
        z_total = z_common + z_disease + z_disease
        feat_ch = base_ch * 4  # must match encoder

        self.net = nn.Sequential(
            nn.Conv2d(z_total, feat_ch, 1),           # project z → feat channels
            ConvTransposeBlock(feat_ch,     base_ch * 2),  # 8→16
            ConvTransposeBlock(base_ch * 2, base_ch),       # 16→32
            ConvTransposeBlock(base_ch,     base_ch),       # 32→64
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z_c, z_d1, z_d2):
        z = torch.cat([z_c, z_d1, z_d2], dim=1)   # (B, z_total, H, W)
        return self.net(z)


class CNNSepVAE(nn.Module):

    def __init__(
        self,
        in_ch:      int = 1,
        base_ch:    int = 32,
        z_common:   int = 4,
        z_disease:  int = 2,
        use_label_attn: bool = False,
        attn_query_dim: int  = 64,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_ch, base_ch, z_common, z_disease,
                                  use_label_attn, attn_query_dim)
        self.decoder = CNNDecoder(in_ch, base_ch, z_common, z_disease)

    def forward(self, x, label_d1=None, label_d2=None):
        enc = self.encoder(x, label_d1, label_d2)
        z_c  = reparameterize(enc['mu_c'],  enc['lv_c'])
        z_d1 = reparameterize(enc['mu_d1'], enc['lv_d1'])
        z_d2 = reparameterize(enc['mu_d2'], enc['lv_d2'])
        x_hat = self.decoder(z_c, z_d1, z_d2)
        return x_hat, enc
