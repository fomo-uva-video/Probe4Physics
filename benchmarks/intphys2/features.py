from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks.intphys2.data import (
    derive_sample_id,
    load_intphys2_rows,
    normalize_intphys2_row,
    resolve_video_path,
)
from models import create_adapter
from models.cache_metadata import resolve_backbone_cache_metadata


class FeatureConfigError(ValueError):
    pass


class FeatureCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureCachePaths:
    cache_dir: Path
    manifest_path: Path
    index_path: Path
    pooled_path: Path
    tokens_path: Path
    signature: str


def run_intphys2_feature_extraction(config: dict[str, Any]) -> dict[str, Any]:
    _validate_extract_config(config)

    paths = resolve_expected_feature_cache_paths(config)
    feature_cfg = _feature_cfg(config)
    warning_log_path = _resolve_warning_log_path(config)

    if _is_valid_cache(paths) and not feature_cfg["force_reextract"]:
        return {
            "feature_cache_dir": str(paths.cache_dir),
            "signature": paths.signature,
            "skipped": True,
            "reason": "valid_cache_exists",
            "warning_log": str(warning_log_path),
        }

    videos_root = Path(str(config["videos_root"])).expanduser()
    _preflight_video_root(videos_root, warning_log_path)

    metadata_file = Path(str(config["metadata_file"]))
    if not metadata_file.exists():
        raise FeatureConfigError(
            f"IntPhys2 metadata file not found: {metadata_file}. "
            "Run init.intphys2 first or point metadata_file to the correct path."
        )

    split_dir = _resolve_split_dir(config)
    split_manifest_path = split_dir / "manifest.json"
    split_scenes_path = split_dir / "split_scenes.parquet"

    if not split_manifest_path.exists() or not split_scenes_path.exists():
        raise FeatureConfigError(
            "Split artifacts not found. Run init first with: python run.py init.intphys2\n"
            f"Missing: {split_scenes_path} or {split_manifest_path}"
        )

    split_manifest = _load_json(split_manifest_path)
    split_scene_rows = _load_split_scenes(split_scenes_path)

    metadata_sha = _sha256_file(metadata_file)
    if metadata_sha != str(split_manifest.get("metadata_sha256", "")):
        raise FeatureConfigError(
            "Metadata hash mismatch with split manifest while extracting features. "
            "Re-run init.intphys2 or restore matching metadata file."
        )

    split_names = feature_cfg["split_names"]
    ordered_records = _ordered_sample_records(
        config,
        metadata_file,
        split_scene_rows,
        split_names,
        warning_log_path=warning_log_path,
    )
    if not ordered_records:
        raise FeatureConfigError(f"No samples found for split_names={split_names}")

    backbone_name, backbone_kwargs = _backbone_cfg(config)
    adapter = create_adapter(backbone_name, **backbone_kwargs)

    decode_cfg = _decode_cfg(config, adapter)
    requested_layer_ids = feature_cfg["layer_ids"] if feature_cfg["layer_ids"] else None

    pooled_acc: dict[int, list[torch.Tensor]] = {}
    tokens_acc: dict[int, list[torch.Tensor]] = {}
    index_rows: list[dict[str, Any]] = []

    selected_layers: tuple[int, ...] | None = None
    adapter_metadata: dict[str, Any] | None = None

    for feature_index, record in enumerate(ordered_records):
        clip = _decode_video_clip(
            video_path=record["video_path"],
            num_frames=decode_cfg["num_frames"],
            crop_size=decode_cfg["crop_size"],
        )

        with torch.no_grad():
            features = adapter.extract(clip, layer_ids=requested_layer_ids)

        if selected_layers is None:
            selected_layers = features.selected_layers
            adapter_metadata = dict(features.metadata)
        elif tuple(features.selected_layers) != tuple(selected_layers):
            raise FeatureCacheError(
                "Adapter returned inconsistent selected_layers during extraction: "
                f"first={selected_layers}, current={features.selected_layers}"
            )

        for layer in features.selected_layers:
            if feature_cfg["include_pooled"]:
                pooled_acc.setdefault(int(layer), []).append(
                    features.pooled_by_layer[layer].detach().cpu()
                )
            if feature_cfg["include_tokens"]:
                tokens_acc.setdefault(int(layer), []).append(
                    features.tokens_by_layer[layer].detach().cpu()
                )

        index_rows.append(
            {
                "feature_index": feature_index,
                "sample_id": record["sample_id"],
                "scene_id": record["scene_id"],
                "split": record["split"],
                "plausibility": int(record["plausibility"]),
                "condition": record["condition"],
                "video_ref": record["video_ref"],
                "video_path": str(record["video_path"]),
            }
        )

    if selected_layers is None:
        raise FeatureCacheError("Feature extraction produced no outputs.")

    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_index(paths.index_path, index_rows)

    if feature_cfg["include_pooled"]:
        pooled_by_layer = {
            layer: torch.cat(values, dim=0)
            for layer, values in sorted(pooled_acc.items())
        }
        torch.save(
            {"selected_layers": list(selected_layers), "by_layer": pooled_by_layer},
            paths.pooled_path,
        )

    if feature_cfg["include_tokens"]:
        tokens_by_layer = {
            layer: torch.cat(values, dim=0)
            for layer, values in sorted(tokens_acc.items())
        }
        torch.save(
            {"selected_layers": list(selected_layers), "by_layer": tokens_by_layer},
            paths.tokens_path,
        )

    manifest = {
        "version": 1,
        "kind": "intphys2_feature_cache",
        "signature": paths.signature,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "split": {
            "dir": str(split_dir),
            "manifest": str(split_manifest_path),
            "manifest_sha256": _sha256_file(split_manifest_path),
            "metadata_sha256": str(split_manifest.get("metadata_sha256", "")),
            "names": split_names,
        },
        "metadata": {
            "file": str(metadata_file),
            "sha256": metadata_sha,
        },
        "backbone": {
            "name": backbone_name,
            "kwargs": backbone_kwargs,
            "metadata": adapter_metadata or {},
        },
        "decode": decode_cfg,
        "features": {
            "include_pooled": bool(feature_cfg["include_pooled"]),
            "include_tokens": bool(feature_cfg["include_tokens"]),
            "selected_layers": list(selected_layers),
            "n_samples": len(index_rows),
            "sample_ids_sha256": _sha256_lines([row["sample_id"] for row in index_rows]),
        },
        "files": {
            "index": paths.index_path.name,
            "pooled": paths.pooled_path.name if feature_cfg["include_pooled"] else "",
            "tokens": paths.tokens_path.name if feature_cfg["include_tokens"] else "",
        },
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "feature_cache_dir": str(paths.cache_dir),
        "signature": paths.signature,
        "skipped": False,
        "n_samples": len(index_rows),
        "selected_layers": list(selected_layers),
        "warning_log": str(warning_log_path),
    }


def resolve_expected_feature_cache_paths(config: dict[str, Any]) -> FeatureCachePaths:
    split_dir = _resolve_split_dir(config)
    split_manifest_path = split_dir / "manifest.json"
    if not split_manifest_path.exists():
        raise FileNotFoundError(
            "Split manifest not found. Run init first with `python run.py init.intphys2`. "
            f"Missing: {split_manifest_path}"
        )

    split_manifest = _load_json(split_manifest_path)
    backbone_name, backbone_kwargs = _backbone_cfg(config)
    feature_cfg = _feature_cfg(config)
    backbone_metadata = resolve_backbone_cache_metadata(backbone_name, backbone_kwargs)

    payload = {
        "metadata_file": str(config.get("metadata_file", "")),
        "backbone_name": backbone_name,
        "backbone_kwargs": backbone_kwargs,
        "backbone_resolved": backbone_metadata,
        "decode": _signature_decode_cfg(config, backbone_metadata),
        "feature_cache": {
            "split_names": feature_cfg["split_names"],
            "layer_ids": feature_cfg["layer_ids"],
            "resolved_layer_ids": (
                feature_cfg["layer_ids"]
                if feature_cfg["layer_ids"]
                else list(backbone_metadata.get("selected_layers", []))
            ),
            "include_pooled": bool(feature_cfg["include_pooled"]),
            "include_tokens": bool(feature_cfg["include_tokens"]),
        },
        "split_dir": str(split_dir),
        "split_manifest_metadata_sha256": str(split_manifest.get("metadata_sha256", "")),
    }
    signature = _sha256_json(payload)[:16]

    variant = str(backbone_metadata.get("variant") or backbone_kwargs.get("variant", "")).strip()
    backbone_id = f"{backbone_name}_{variant}" if variant else str(backbone_name)
    split_key = "-".join(feature_cfg["split_names"])

    base_dir = Path(feature_cfg["dir"]).expanduser().resolve()
    cache_dir = base_dir / backbone_id / split_key / signature

    return FeatureCachePaths(
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        index_path=cache_dir / "index.parquet",
        pooled_path=cache_dir / "features_pooled.pt",
        tokens_path=cache_dir / "features_tokens.pt",
        signature=signature,
    )


def has_valid_feature_cache(config: dict[str, Any]) -> bool:
    try:
        paths = resolve_expected_feature_cache_paths(config)
    except Exception:
        return False
    return _is_valid_cache(paths)


def load_feature_cache_for_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = resolve_expected_feature_cache_paths(config)
    if not _is_valid_cache(paths):
        raise FeatureCacheError(
            "Feature cache is missing or invalid for current config. "
            "Run `python run.py extract.intphys2` first."
        )

    manifest = _load_json(paths.manifest_path)
    frame = _load_index(paths.index_path)

    pooled_payload: dict[str, Any] | None = None
    tokens_payload: dict[str, Any] | None = None

    if str(manifest.get("files", {}).get("pooled", "")):
        pooled_payload = torch.load(str(paths.pooled_path), map_location="cpu", weights_only=False)
    if str(manifest.get("files", {}).get("tokens", "")):
        tokens_payload = torch.load(str(paths.tokens_path), map_location="cpu", weights_only=False)

    _validate_loaded_feature_payloads(
        manifest=manifest,
        index=frame,
        pooled=pooled_payload,
        tokens=tokens_payload,
    )

    return {
        "paths": paths,
        "manifest": manifest,
        "index": frame,
        "pooled": pooled_payload,
        "tokens": tokens_payload,
    }


def _validate_extract_config(config: dict[str, Any]) -> None:
    required = ["metadata_file", "videos_root", "cache_dir", "split"]
    missing = [key for key in required if key not in config]
    if missing:
        raise FeatureConfigError(f"Missing config keys: {missing}")


def _resolve_split_dir(config: dict[str, Any]) -> Path:
    raw_split = config.get("split", {})
    if not isinstance(raw_split, dict):
        raise FeatureConfigError("split config must be a dictionary")
    return Path(str(raw_split.get("dir", "data/splits/intphys2")))


def _feature_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("feature_cache", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise FeatureConfigError("feature_cache must be a dictionary")

    split_names = raw.get("split_names", ["train", "val", "test"])
    if not isinstance(split_names, (list, tuple)) or not split_names:
        raise FeatureConfigError("feature_cache.split_names must be a non-empty list")

    layer_ids_raw = raw.get("layer_ids", [])
    if layer_ids_raw is None:
        layer_ids_raw = []
    if not isinstance(layer_ids_raw, (list, tuple)):
        raise FeatureConfigError("feature_cache.layer_ids must be a list")

    include_pooled = bool(raw.get("include_pooled", True))
    include_tokens = bool(raw.get("include_tokens", True))
    if not include_pooled and not include_tokens:
        raise FeatureConfigError(
            "feature_cache.include_pooled and feature_cache.include_tokens cannot both be false"
        )

    return {
        "dir": str(raw.get("dir", "artifacts/features/intphys2")),
        "split_names": [str(item) for item in split_names],
        "layer_ids": [int(v) for v in layer_ids_raw],
        "include_pooled": include_pooled,
        "include_tokens": include_tokens,
        "force_reextract": bool(raw.get("force_reextract", False)),
    }


def _backbone_cfg(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = config.get("backbone", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise FeatureConfigError("backbone must be a dictionary")

    name = str(raw.get("name", "jepa_v1")).strip()
    kwargs = raw.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise FeatureConfigError("backbone.kwargs must be a dictionary")
    return name, dict(kwargs)


def _decode_cfg(config: dict[str, Any], adapter: Any) -> dict[str, Any]:
    raw = config.get("decode", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise FeatureConfigError("decode must be a dictionary")

    default_frames = int(getattr(adapter, "frames_per_clip", 16))
    default_crop = int(getattr(adapter, "crop_size", 224))

    return {
        "num_frames": int(raw.get("num_frames", default_frames)),
        "sampling": str(raw.get("sampling", "uniform")),
        "crop_size": int(raw.get("crop_size", default_crop)),
    }


def _signature_decode_cfg(
    config: dict[str, Any],
    backbone_metadata: dict[str, Any],
) -> dict[str, Any]:
    raw = config.get("decode", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise FeatureConfigError("decode must be a dictionary")

    default_frames = int(backbone_metadata.get("frames_per_clip") or 16)
    default_crop = int(backbone_metadata.get("crop_size") or 224)
    return {
        "num_frames": int(raw.get("num_frames", default_frames)),
        "sampling": str(raw.get("sampling", "uniform")),
        "crop_size": int(raw.get("crop_size", default_crop)),
    }


def _ordered_sample_records(
    config: dict[str, Any],
    metadata_file: Path,
    split_scene_rows: list[dict[str, Any]],
    split_names: list[str],
    *,
    warning_log_path: Path,
) -> list[dict[str, Any]]:
    ordered_ids_by_split: dict[str, list[str]] = {}
    for split_name in split_names:
        scene_rows = [row for row in split_scene_rows if str(row.get("split", "")) == split_name]
        if not scene_rows:
            continue

        ordered_sample_ids: list[str] = []
        for row in sorted(scene_rows, key=lambda item: str(item.get("scene_id", ""))):
            ordered_sample_ids.extend(_parse_sample_ids_json(row.get("sample_ids_json", "[]")))
        ordered_ids_by_split[split_name] = ordered_sample_ids

    if not ordered_ids_by_split:
        return []

    all_rows = load_intphys2_rows(metadata_file)
    norm_rows = [normalize_intphys2_row(row) for row in all_rows]

    row_by_sample: dict[str, dict[str, Any]] = {}
    target_ids = {
        sample_id
        for sample_ids in ordered_ids_by_split.values()
        for sample_id in sample_ids
    }

    for norm_row in norm_rows:
        sample_id = derive_sample_id(norm_row)
        if sample_id in target_ids and sample_id not in row_by_sample:
            row_by_sample[sample_id] = norm_row

    missing = [sid for sid in target_ids if sid not in row_by_sample]
    if missing:
        raise FeatureConfigError(
            "Split references sample_ids missing from metadata file. "
            f"First missing ids: {sorted(missing)[:5]}"
        )

    ordered_records: list[dict[str, Any]] = []

    for split_name in split_names:
        sample_ids = ordered_ids_by_split.get(split_name, [])
        if not sample_ids:
            continue

        for sample_id in sample_ids:
            norm_row = row_by_sample[sample_id]
            video_ref = str(norm_row.get("video_path", "")).strip()

            resolved_video = resolve_video_path(
                video_ref=video_ref,
                videos_root=config["videos_root"],
                cache_dir=config["cache_dir"],
                materialize_missing=bool(config.get("materialize_missing", False)),
                timeout_seconds=int(config.get("download_timeout_seconds", 120)),
            )

            video_path = Path(resolved_video)
            if not video_path.exists() or not video_path.is_file():
                _append_warning(
                    warning_log_path,
                    warning_type="missing_video_file",
                    message="Resolved video path does not exist for extraction.",
                    details={
                        "sample_id": sample_id,
                        "video_ref": video_ref,
                        "resolved_path": str(video_path),
                        "videos_root": str(config["videos_root"]),
                    },
                )
                raise FileNotFoundError(
                    "Resolved video path does not exist for extraction: "
                    f"sample_id={sample_id}, path={video_path}. "
                    f"Warning log: {warning_log_path}"
                )

            ordered_records.append(
                {
                    "sample_id": sample_id,
                    "scene_id": str(norm_row.get("scene_id", "")),
                    "split": split_name,
                    "plausibility": int(norm_row.get("plausibility", -1)),
                    "condition": str(norm_row.get("condition", "")),
                    "video_ref": video_ref,
                    "video_path": str(video_path),
                }
            )

    return ordered_records


def _decode_video_clip(video_path: str, num_frames: int, crop_size: int) -> torch.Tensor:
    try:
        import av
        import numpy as np
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Feature extraction requires av, numpy, and torch. "
            "Install dependencies from environment.yml."
        ) from exc

    if num_frames <= 0:
        raise FeatureConfigError(f"decode.num_frames must be > 0, got {num_frames}")
    if crop_size <= 0:
        raise FeatureConfigError(f"decode.crop_size must be > 0, got {crop_size}")

    container = av.open(video_path)
    try:
        frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
    finally:
        container.close()

    if not frames:
        raise FeatureCacheError(f"No decoded frames in video: {video_path}")

    indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
    sampled = [frames[int(i)] for i in indices]
    array = np.stack(sampled, axis=0)  # [T, H, W, C]

    clip = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous().float() / 255.0  # [C, T, H, W]
    clip = clip.unsqueeze(0)  # [B, C, T, H, W]

    b, c, t, h, w = clip.shape
    if h != crop_size or w != crop_size:
        frames_4d = clip.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        resized = F.interpolate(
            frames_4d, size=(crop_size, crop_size), mode="bilinear", align_corners=False
        )
        clip = resized.reshape(b, t, c, crop_size, crop_size).permute(0, 2, 1, 3, 4).contiguous()

    return clip


def _load_split_scenes(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading split_scenes.parquet requires pandas. "
            "Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Writing index.parquet requires pandas. Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _load_index(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading index.parquet requires pandas. Install it with: python -m pip install pandas"
        ) from exc

    return pd.read_parquet(path)


def _validate_loaded_feature_payloads(
    *,
    manifest: dict[str, Any],
    index: Any,
    pooled: dict[str, Any] | None,
    tokens: dict[str, Any] | None,
) -> None:
    n_rows = len(index)
    selected_layers = [int(layer) for layer in manifest.get("features", {}).get("selected_layers", [])]

    if pooled is not None:
        _validate_feature_payload(
            payload=pooled,
            name="pooled",
            selected_layers=selected_layers,
            n_rows=n_rows,
            expected_rank=2,
        )
    if tokens is not None:
        _validate_feature_payload(
            payload=tokens,
            name="tokens",
            selected_layers=selected_layers,
            n_rows=n_rows,
            expected_rank=3,
        )


def _validate_feature_payload(
    *,
    payload: dict[str, Any],
    name: str,
    selected_layers: list[int],
    n_rows: int,
    expected_rank: int,
) -> None:
    payload_layers = [int(layer) for layer in payload.get("selected_layers", [])]
    if selected_layers and payload_layers and payload_layers != selected_layers:
        raise FeatureCacheError(
            f"{name} feature payload layers {payload_layers} do not match manifest layers {selected_layers}. "
            "Re-run `python run.py extract.intphys2`."
        )

    by_layer = payload.get("by_layer", {})
    if not isinstance(by_layer, dict):
        raise FeatureCacheError(f"{name} feature payload missing by_layer mapping. Re-run extraction.")

    required_layers = selected_layers or payload_layers
    for layer in required_layers:
        if layer not in by_layer:
            raise FeatureCacheError(
                f"{name} feature payload missing layer {layer}. Re-run `python run.py extract.intphys2`."
            )
        tensor = by_layer[layer]
        if not isinstance(tensor, torch.Tensor):
            raise FeatureCacheError(f"{name} layer {layer} is not a tensor. Re-run extraction.")
        if tensor.ndim != expected_rank:
            raise FeatureCacheError(
                f"{name} layer {layer} expected rank {expected_rank}, got shape {tuple(tensor.shape)}. "
                "Re-run extraction."
            )
        if int(tensor.shape[0]) != int(n_rows):
            raise FeatureCacheError(
                f"{name} layer {layer} row count {int(tensor.shape[0])} does not match index rows {n_rows}. "
                "Re-run extraction."
            )


def _is_valid_cache(paths: FeatureCachePaths) -> bool:
    if not paths.manifest_path.exists() or not paths.index_path.exists():
        return False

    try:
        manifest = _load_json(paths.manifest_path)
    except Exception:
        return False

    if str(manifest.get("signature", "")) != paths.signature:
        return False

    files_cfg = manifest.get("files", {})
    if str(files_cfg.get("pooled", "")) and not paths.pooled_path.exists():
        return False
    if str(files_cfg.get("tokens", "")) and not paths.tokens_path.exists():
        return False

    return True


def _parse_sample_ids_json(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    parsed = json.loads(str(raw))
    if not isinstance(parsed, list):
        raise FeatureConfigError(f"sample_ids_json must decode to a list, got: {parsed!r}")
    return [str(item) for item in parsed]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_warning_log_path(config: dict[str, Any]) -> Path:
    raw = config.get("feature_cache", {})
    if not isinstance(raw, dict):
        raw = {}
    root = Path(str(raw.get("dir", "artifacts/features/intphys2"))).expanduser().resolve()
    return root / "extract_warnings.log"


def _preflight_video_root(videos_root: Path, warning_log_path: Path) -> None:
    if not videos_root.exists():
        _append_warning(
            warning_log_path,
            warning_type="missing_videos_root",
            message="videos_root does not exist before extraction.",
            details={"videos_root": str(videos_root)},
        )
        raise FileNotFoundError(
            f"videos_root does not exist: {videos_root}. "
            f"Download or mount IntPhys2 videos first. Warning log: {warning_log_path}"
        )

    if not videos_root.is_dir():
        _append_warning(
            warning_log_path,
            warning_type="invalid_videos_root",
            message="videos_root is not a directory.",
            details={"videos_root": str(videos_root)},
        )
        raise FileNotFoundError(
            f"videos_root is not a directory: {videos_root}. "
            f"Warning log: {warning_log_path}"
        )

    if not _contains_any_video(videos_root):
        _append_warning(
            warning_log_path,
            warning_type="empty_videos_root",
            message="videos_root exists but no video files were found.",
            details={"videos_root": str(videos_root)},
        )
        raise FileNotFoundError(
            f"No video files found under videos_root: {videos_root}. "
            f"Warning log: {warning_log_path}"
        )


def _contains_any_video(videos_root: Path) -> bool:
    patterns = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
    for pattern in patterns:
        if next(videos_root.rglob(pattern), None) is not None:
            return True
    return False


def _append_warning(
    warning_log_path: Path,
    *,
    warning_type: str,
    message: str,
    details: dict[str, Any],
) -> None:
    warning_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "warning_type": str(warning_type),
        "message": str(message),
        "details": details,
    }
    with warning_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
