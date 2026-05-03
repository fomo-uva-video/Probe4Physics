from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from probes.mlp import MLPProbe
from probes.temporal_attn import (
    TemporalAttentiveProbe,
    _MultiheadCrossAttention,
    _MultiheadSelfAttention,
)


def _manual_self_attention(module: _MultiheadSelfAttention, x: torch.Tensor) -> torch.Tensor:
    embed_dim = module.mha.embed_dim
    num_heads = module.mha.num_heads
    head_dim = embed_dim // num_heads
    scale = head_dim**0.5
    batch_size, seq_len, _ = x.shape

    qkv = F.linear(x, module.mha.in_proj_weight, module.mha.in_proj_bias)
    q, k, v = qkv.split(embed_dim, dim=-1)
    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    return F.linear(out, module.mha.out_proj.weight, module.mha.out_proj.bias)


def _manual_cross_attention(
    module: _MultiheadCrossAttention,
    query: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    embed_dim = module.mha.embed_dim
    num_heads = module.mha.num_heads
    head_dim = embed_dim // num_heads
    scale = head_dim**0.5
    batch_size, query_len, _ = query.shape
    context_len = context.shape[1]

    q_weight, k_weight, v_weight = module.mha.in_proj_weight.split(embed_dim, dim=0)
    q_bias, k_bias, v_bias = module.mha.in_proj_bias.split(embed_dim, dim=0)
    q = F.linear(query, q_weight, q_bias)
    k = F.linear(context, k_weight, k_bias)
    v = F.linear(context, v_weight, v_bias)

    q = q.reshape(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, context_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, context_len, num_heads, head_dim).transpose(1, 2)

    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).reshape(batch_size, query_len, embed_dim)
    return F.linear(out, module.mha.out_proj.weight, module.mha.out_proj.bias)
class MLPProbeTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        probe = MLPProbe(
            input_dim=4,
            num_classes=2,
            hidden_dims=[16, 8],
            dropout=0.0,
            device="cpu",
        )

        x = torch.randn(64, 4)
        y = (x[:, 0] > 0).long()
        fit = probe.fit(x, y, epochs=3, lr=1e-2, batch_size=16, seed=7)

        self.assertEqual(fit.n_epochs, 3)
        self.assertGreaterEqual(fit.train_loss, 0.0)
        self.assertGreaterEqual(fit.train_accuracy, 0.0)
        self.assertLessEqual(fit.train_accuracy, 100.0)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mlp.pt"
            probe.save(ckpt, metadata={"tag": "unit-test"})

            loaded = MLPProbe.load(ckpt, device="cpu")
            pred = loaded.predict(x)
            self.assertEqual(tuple(pred.shape), (64,))


class TemporalAttentiveProbeTests(unittest.TestCase):
    def test_attention_modules_match_legacy_attention_math(self) -> None:
        torch.manual_seed(7)

        self_attn = _MultiheadSelfAttention(embed_dim=8, num_heads=2, dropout=0.0)
        cross_attn = _MultiheadCrossAttention(embed_dim=8, num_heads=2, dropout=0.0)
        self_attn.eval()
        cross_attn.eval()

        x = torch.randn(3, 5, 8)
        query = torch.randn(3, 1, 8)
        context = torch.randn(3, 5, 8)

        torch.testing.assert_close(self_attn(x), _manual_self_attention(self_attn, x))
        torch.testing.assert_close(
            cross_attn(query, context),
            _manual_cross_attention(cross_attn, query, context),
        )

    def test_save_and_load_roundtrip(self) -> None:
        probe = TemporalAttentiveProbe(
            input_dim=8,
            num_classes=2,
            embed_dim=8,
            num_heads=2,
            num_self_attn_blocks=1,
            mlp_ratio=2.0,
            dropout=0.0,
            device="cpu",
        )

        x = torch.randn(32, 6, 8)
        y = (x[:, :, 0].mean(dim=1) > 0).long()
        fit = probe.fit(x, y, epochs=2, lr=1e-2, batch_size=8, seed=7)

        self.assertEqual(fit.n_epochs, 2)
        self.assertGreaterEqual(fit.train_loss, 0.0)
        self.assertGreaterEqual(fit.train_accuracy, 0.0)
        self.assertLessEqual(fit.train_accuracy, 100.0)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "temporal_attn.pt"
            probe.save(ckpt, metadata={"tag": "unit-test"})

            loaded = TemporalAttentiveProbe.load(ckpt, device="cpu")
            pred = loaded.predict(x)
            self.assertEqual(tuple(pred.shape), (32,))

    def test_predict_logits_supports_input_projection(self) -> None:
        probe = TemporalAttentiveProbe(
            input_dim=6,
            num_classes=3,
            embed_dim=8,
            num_heads=2,
            num_self_attn_blocks=1,
            mlp_ratio=2.0,
            dropout=0.0,
            device="cpu",
        )

        logits = probe.predict_logits(torch.randn(7, 4, 6))
        self.assertEqual(tuple(logits.shape), (7, 3))

    def test_fit_rejects_non_token_input(self) -> None:
        probe = TemporalAttentiveProbe(
            input_dim=8,
            num_classes=2,
            embed_dim=8,
            num_heads=2,
            num_self_attn_blocks=1,
            mlp_ratio=2.0,
            dropout=0.0,
            device="cpu",
        )

        x_bad = torch.randn(32, 8)
        y = torch.randint(0, 2, (32,), dtype=torch.long)

        with self.assertRaises(ValueError):
            probe.fit(x_bad, y, epochs=1)

    def test_fit_supports_batched_validation(self) -> None:
        probe = TemporalAttentiveProbe(
            input_dim=8,
            num_classes=2,
            embed_dim=8,
            num_heads=2,
            num_self_attn_blocks=1,
            mlp_ratio=2.0,
            dropout=0.0,
            device="cpu",
        )

        x = torch.randn(24, 5, 8)
        y = (x[:, :, 0].mean(dim=1) > 0).long()

        fit = probe.fit(
            x,
            y,
            x_val=x,
            y_val=y,
            epochs=2,
            lr=1e-2,
            batch_size=6,
            eval_batch_size=4,
            seed=7,
        )

        self.assertEqual(fit.n_epochs, 2)
        self.assertEqual(len(fit.history), 2)
        self.assertTrue(all("val_loss" in row for row in fit.history))
        self.assertTrue(all("val_accuracy" in row for row in fit.history))
        self.assertIsNotNone(fit.best_epoch)
        self.assertIsNotNone(fit.best_val_accuracy)


if __name__ == "__main__":
    unittest.main()
