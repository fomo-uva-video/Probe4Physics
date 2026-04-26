from __future__ import annotations

from typing import Any

import torch
from torch import nn

from probes.base import BaseClassifierProbe
from probes.registry import register_probe


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _MultiheadSelfAttention(nn.Module):
    """PyTorch-backed multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


class _MultiheadCrossAttention(nn.Module):
    """PyTorch-backed multi-head cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(query, context, context, need_weights=False)
        return out


class _TransformerBlock(nn.Module):
    """Standard transformer block: (pre-norm) self-attention + MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _MultiheadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _CrossAttentionBlock(nn.Module):
    """Final transformer block with cross-attention over a learnable query token.

    Architecture (as specified in the proposal):
    - Cross-attention between learnable query and the token sequence, with the
      cross-attention output added back to the query as a residual.
    - LayerNorm applied to the updated query.
    - MLP with a single GeLU activation applied to the normalised query.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attn = _MultiheadCrossAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query   : [B, 1, D] – the learnable query token (one per sample)
        context : [B, T, D] – sequence of frozen encoder tokens

        Returns
        -------
        Tensor of shape [B, 1, D]
        """
        # Cross-attention output added to query as residual
        query = query + self.cross_attn(query, context)
        # LayerNorm → MLP (no second residual in query path, matching proposal text)
        query = query + self.mlp(self.norm(query))
        return query


class _AttentiveProbeModel(nn.Module):
    """The full attentive probe neural network.

    Three self-attention transformer blocks followed by one cross-attention
    block (with a learnable query token), then a linear classifier.

    Parameters
    ----------
    embed_dim :
        Dimension of the token features (D). An optional input projection is
        added when ``input_dim != embed_dim``.
    input_dim :
        Actual dimension of incoming feature tokens.  If ``input_dim !=
        embed_dim``, a learned linear projection is prepended.
    num_heads :
        Number of attention heads in every block (default 16).
    num_classes :
        Number of output classes.
    num_self_attn_blocks :
        Number of self-attention blocks before the cross-attention block
        (default 3).
    mlp_ratio :
        Hidden-dim multiplier for the MLP inside each block (default 4.0).
    dropout :
        Attention dropout probability (default 0.0).
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
        num_self_attn_blocks: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Optional input projection when input_dim ≠ embed_dim
        if input_dim != embed_dim:
            self.input_proj: nn.Module = nn.Linear(input_dim, embed_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Self-attention blocks (first 3)
        self.self_attn_blocks = nn.ModuleList(
            [
                _TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_self_attn_blocks)
            ]
        )

        # Cross-attention block (final block)
        self.cross_attn_block = _CrossAttentionBlock(
            embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )

        # Learnable query token [1, 1, D]
        self.query_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)

        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape [B, T, D_in]
            Batch of token sequences from the frozen encoder.

        Returns
        -------
        Tensor of shape [B, num_classes]
        """
        B = x.shape[0]

        # Project input if needed
        x = self.input_proj(x)  # [B, T, embed_dim]

        # Self-attention blocks
        for block in self.self_attn_blocks:
            x = block(x)

        # Expand learnable query to batch dimension
        query = self.query_token.expand(B, -1, -1)  # [B, 1, embed_dim]

        # Cross-attention block → [B, 1, embed_dim]
        query = self.cross_attn_block(query, x)

        # Squeeze the sequence dimension and classify
        return self.classifier(query.squeeze(1))  # [B, num_classes]

# ---------------------------------------------------------------------------
# Public probe class
# ---------------------------------------------------------------------------


class TemporalAttentiveProbe(BaseClassifierProbe):
    """Attentive probe over token sequences from a frozen video encoder.

    The architecture follows the proposal specification:
    - Three standard self-attention transformer blocks.
    - One cross-attention block with a learnable query token.
    - The cross-attention output is added to the query as a residual, followed
      by LayerNorm and an MLP with a single GeLU activation.
    - A final linear classifier head.

    All blocks use 16 attention heads by default and an MLP with hidden
    dimension 4× the embedding dimension.

    Input tokens are expected to have shape ``[N, T, D]`` (N samples, T tokens,
    D feature dimension).  An optional linear projection is applied when the
    feature dimension differs from the internal ``embed_dim``.

    Parameters
    ----------
    input_dim :
        Dimension of each input token (D).
    num_classes :
        Number of output classes (must be ≥ 2).
    embed_dim :
        Internal embedding dimension of the transformer. Defaults to
        ``input_dim`` (no projection needed in that case).
    num_heads :
        Number of attention heads in every block (default 16).
    num_self_attn_blocks :
        Number of self-attention blocks before the cross-attention block
        (default 3, as per the proposal).
    mlp_ratio :
        Hidden-dimension multiplier for the feed-forward MLP inside each
        block (default 4.0).
    dropout :
        Attention dropout probability (default 0.0).
    device :
        PyTorch device string or object.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        embed_dim: int | None = None,
        num_heads: int = 16,
        num_self_attn_blocks: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        resolved_input_dim = int(input_dim)
        self.embed_dim = int(embed_dim) if embed_dim is not None else resolved_input_dim
        self.num_heads = int(num_heads)
        self.num_self_attn_blocks = int(num_self_attn_blocks)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        super().__init__(input_dim=resolved_input_dim, num_classes=num_classes, device=device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        return _AttentiveProbeModel(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            num_self_attn_blocks=self.num_self_attn_blocks,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
        )

    def _checkpoint_type(self) -> str:
        return "temporal_attn"

    def _checkpoint_extra_fields(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_self_attn_blocks": self.num_self_attn_blocks,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        }

    @classmethod
    def _init_kwargs_from_payload(
        cls,
        payload: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        probe_type = str(payload.get("type", ""))
        if probe_type != "temporal_attn":
            raise ValueError(
                f"Checkpoint type mismatch: expected 'temporal_attn', got '{probe_type}'"
            )
        return {
            "input_dim": int(payload["input_dim"]),
            "num_classes": int(payload["num_classes"]),
            "embed_dim": int(payload.get("embed_dim", payload["input_dim"])),
            "num_heads": int(payload.get("num_heads", 16)),
            "num_self_attn_blocks": int(payload.get("num_self_attn_blocks", 3)),
            "mlp_ratio": float(payload.get("mlp_ratio", 4.0)),
            "dropout": float(payload.get("dropout", 0.0)),
            "device": device,
        }

    def _validate_and_cast_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"TemporalAttentiveProbe expects token tensors with shape [N, T, D], got {tuple(x.shape)}"
            )
        return x.to(dtype=torch.float32)


def create_temporal_attn_probe(**kwargs: Any) -> TemporalAttentiveProbe:
    return TemporalAttentiveProbe(**kwargs)


register_probe("temporal_attn", create_temporal_attn_probe, replace=True)
