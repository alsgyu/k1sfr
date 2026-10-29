"""Torch-only VIGOR networks (no rsl_rl / Isaac imports, TorchScript-compatible).

Teacher actor : a = pi_theta(z_g, o_t),            z_g = F(k_t, h_t)
Student       : a = pi_phi(z~_g, o_t),             z~_g = G(o_{t-T:t} [, I_{t-k:t}])
Both latents are normalised to sqrt(d) * z / ||z|| (paper Sec. III-B).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU, "tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    dims = [in_dim] + list(hidden)
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), act()]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def normalize_latent(z: torch.Tensor, scale: float) -> torch.Tensor:
    """sqrt(d) * z / ||z||; ``scale`` = sqrt(d) is passed in so tracing keeps it constant."""
    return scale * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class _InputSpec(nn.Module):
    """Lets Isaac Lab's ONNX exporter read ``actor[0].in_features``."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features


class GoalInContextActor(nn.Module):
    """obs = [o_t (proprio) | k_t (keyframes) | h_t (terrain)] -> actions."""

    def __init__(self, proprio_dim: int, context_dim: int, num_actions: int, latent_dim: int = 32,
                 encoder_hidden_dims: list[int] = [256, 128], actor_hidden_dims: list[int] = [512, 256, 128],
                 activation: str = "elu"):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.context_dim = context_dim
        self.latent_scale = math.sqrt(latent_dim)
        self.encoder = mlp(context_dim, encoder_hidden_dims, latent_dim, activation)
        self.head = mlp(latent_dim + proprio_dim, actor_hidden_dims, num_actions, activation)
        self._spec = _InputSpec(proprio_dim + context_dim)

    def __getitem__(self, idx: int) -> nn.Module:
        if idx != 0:
            raise IndexError(idx)
        return self._spec

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return normalize_latent(self.encoder(obs[:, self.proprio_dim:]), self.latent_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs)
        return self.head(torch.cat((z, obs[:, : self.proprio_dim]), dim=-1))


class ProprioHistoryEncoder(nn.Module):
    """Temporal conv encoder over Isaac Lab's term-major flattened history.

    Isaac Lab flattens each term's history as (T, d_k) row-major, oldest first,
    and concatenates terms; this module undoes that layout.
    """

    def __init__(self, term_dims: list[int], history: int, out_dim: int = 128, activation: str = "elu"):
        super().__init__()
        self.term_dims = term_dims
        self.history = history
        self.frame_dim = sum(term_dims)
        act = _ACTIVATIONS[activation]
        self.conv = nn.Sequential(
            nn.Conv1d(self.frame_dim, 64, kernel_size=5, stride=2), act(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2), act(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, self.frame_dim, history)).numel()
        self.out = nn.Sequential(nn.Linear(n, out_dim), act())

    def frames(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, T * frame_dim) term-major -> (B, T, frame_dim)."""
        parts = []
        start = 0
        for d in self.term_dims:
            parts.append(obs[:, start : start + d * self.history].reshape(-1, self.history, d))
            start += d * self.history
        return torch.cat(parts, dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.frames(obs).transpose(1, 2)
        return self.out(self.conv(x).flatten(1))


class VigorStudent(nn.Module):
    """Deployable student: proprio history (+ optional depth) -> (actions, latent)."""

    def __init__(self, term_dims: list[int], history: int, num_actions: int, latent_dim: int = 32,
                 actor_hidden_dims: list[int] = [512, 256, 128], fusion_hidden_dims: list[int] = [256],
                 activation: str = "elu"):
        super().__init__()
        self.latent_scale = math.sqrt(latent_dim)
        self.history_encoder = ProprioHistoryEncoder(term_dims, history, 128, activation)
        self.fusion = mlp(128, fusion_hidden_dims, latent_dim, activation)
        self.head = mlp(latent_dim + sum(term_dims), actor_hidden_dims, num_actions, activation)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return normalize_latent(self.fusion(self.history_encoder(obs)), self.latent_scale)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(obs)
        current = self.history_encoder.frames(obs)[:, -1]
        return self.head(torch.cat((z, current), dim=-1)), z


class NormalizedActor(nn.Module):
    """Export wrapper: raw observation -> action, normaliser baked in (trace with torch.jit.trace)."""

    def __init__(self, actor: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", mean.detach().clone().reshape(1, -1))
        self.register_buffer("std", std.detach().clone().reshape(1, -1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.actor((obs - self.mean) / self.std)
        return out[0] if isinstance(out, tuple) else out
