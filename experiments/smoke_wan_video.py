from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch' in the active environment.\n"
        "Install with: python -m pip install torch"
    ) from exc

try:
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Missing Wan support in the active diffusers environment.\n"
        "Install with: python -m pip install -U diffusers transformers accelerate safetensors sentencepiece"
    ) from exc

from models import create_adapter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test for the Wan video adapter.")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional variant override from configs/backbones.yaml.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the smoke run (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic batch size.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="1",
        help="Comma-separated probe slots to extract during the smoke (default: 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    adapter_kwargs: dict[str, object] = {"device": args.device}
    if args.variant:
        adapter_kwargs["variant"] = args.variant
    if args.hf_cache_dir:
        adapter_kwargs["hf_cache_dir"] = args.hf_cache_dir

    try:
        adapter = create_adapter("wan_video", **adapter_kwargs)
    except Exception as exc:
        print("Adapter initialization failed.")
        print(str(exc))
        return 1

    layer_ids = [int(item) for item in args.layers.split(",") if item.strip()]
    clips = torch.randn(
        args.batch_size,
        3,
        adapter.frames_per_clip,
        adapter.crop_size,
        adapter.crop_size,
    )

    with torch.no_grad():
        features = adapter.extract(clips, layer_ids=layer_ids)

    print("Adapter:", "wan_video")
    print("Variant:", features.metadata.get("variant"))
    print("Model:", features.metadata.get("model_name"))
    print("HF model ID:", features.metadata.get("hf_model_id"))
    print("Selected layers:", features.selected_layers)
    print("Token shapes:", {k: tuple(v.shape) for k, v in features.tokens_by_layer.items()})
    print("Pooled shapes:", {k: tuple(v.shape) for k, v in features.pooled_by_layer.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
