from __future__ import annotations

from typing import Any

import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESSING_VERSION = 1


def normalize_rgb_imagenet(clips: torch.Tensor) -> torch.Tensor:
    """Normalize raw RGB video clips with ImageNet channel statistics.

    Args:
        clips: Raw float tensor shaped [B, C, T, H, W] with values in [0, 1].
    """

    mean = torch.tensor(IMAGENET_MEAN, dtype=clips.dtype, device=clips.device).view(
        1, 3, 1, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, dtype=clips.dtype, device=clips.device).view(
        1, 3, 1, 1, 1
    )
    return (clips - mean) / std


def imagenet_preprocessing_metadata(*, family: str, layout: str = "BCTHW") -> dict[str, Any]:
    return {
        "version": PREPROCESSING_VERSION,
        "family": str(family),
        "input_color": "rgb",
        "input_range": [0.0, 1.0],
        "layout": layout,
        "normalization": "imagenet_mean_std",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


def ltx_preprocessing_metadata(*, normalize_input: bool) -> dict[str, Any]:
    return {
        "version": PREPROCESSING_VERSION,
        "family": "ltx_video_vae",
        "input_color": "rgb",
        "input_range": [0.0, 1.0],
        "layout": "BCTHW",
        "normalization": "scale_to_minus_one_one" if normalize_input else "identity",
        "output_range": [-1.0, 1.0] if normalize_input else [0.0, 1.0],
    }
