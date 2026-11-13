"""
mlp_vae.py — A0 MLP encoder/decoder.

Used only for E0/E1 on flat latent space.
Single head z of dim z_dim (default 8 = 4+2+2 to match split-head total).
"""
import torch
import torch.nn as nn

from losses.base import reparameterize


class MLPEncoder(nn.Module):

    def __init__(self, in_dim: int = 4096, hidden: int = 512, z_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.mu_head     = nn.Linear(hidden // 2, z_dim)
        self.logvar_head = nn.Linear(hidden // 2, z_dim)

    def forward(self, x: torch.Tensor):
        h  = self.net(x.view(x.shape[0], -1))
        mu = self.mu_head(h)
        lv = self.logvar_head(h).clamp(-10, 2)
        return mu, lv


class MLPDecoder(nn.Module):

    def __init__(self, out_dim: int = 4096, hidden: int = 512, z_dim: int = 8,
                 img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.img_shape = img_shape
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.shape[0], *self.img_shape)


class MLPVAE(nn.Module):

    def __init__(self, img_shape: tuple = (1, 64, 64), z_dim: int = 8):
        super().__init__()
        in_dim = 1
        for d in img_shape:
            in_dim *= d
        self.encoder = MLPEncoder(in_dim, 512, z_dim)
        self.decoder = MLPDecoder(in_dim, 512, z_dim, img_shape)

    def forward(self, x: torch.Tensor):
        mu, lv = self.encoder(x)
        z      = reparameterize(mu, lv)
        x_hat  = self.decoder(z)
        return x_hat, mu, lv
