from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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


@dataclass(frozen=True)
class ResumePaths:
    root_dir: Path
    chunks_dir: Path
    state_path: Path
    events_path: Path
    lock_path: Path


def run_intphys2_feature_extraction(config: dict[str, Any]) -> dict[str, Any]:
    _validate_extract_config(config)

    paths = resolve_expected_feature_cache_paths(config)
    feature_cfg = _feature_cfg(config)
    warning_log_path = _resolve_warning_log_path(config)

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
    ordered_records_all = _ordered_sample_records(
        config,
        metadata_file,
        split_scene_rows,
        split_names,
        warning_log_path=warning_log_path,
    )
    ordered_records = _limit_ordered_records(ordered_records_all, feature_cfg["max_samples"])
    if not ordered_records:
        raise FeatureConfigError(f"No samples found for split_names={split_names}")

    backbone_name, backbone_kwargs = _backbone_cfg(config)
    adapter = create_adapter(backbone_name, **backbone_kwargs)
    decode_cfg = _decode_cfg(config, adapter)
    requested_layer_ids = feature_cfg["layer_ids"] if feature_cfg["layer_ids"] else None

    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    resume_paths = _resume_paths(paths.cache_dir)
    if feature_cfg["force_reextract"]:
        _reset_resume_state(paths, resume_paths)
    if not feature_cfg["resume_enabled"] and resume_paths.root_dir.exists():
        shutil.rmtree(resume_paths.root_dir)

    run_started_utc = datetime.now(tz=timezone.utc).isoformat()
    wall_start = time.perf_counter()
    config_hash = _sha256_json(
        _resume_config_payload(
            config=config,
            feature_cfg=feature_cfg,
            split_manifest=split_manifest,
            split_dir=split_dir,
            metadata_file=str(metadata_file),
            backbone_name=backbone_name,
            backbone_kwargs=backbone_kwargs,
            decode_cfg=decode_cfg,
        )
    )
    ordered_sample_ids_all = [str(row["sample_id"]) for row in ordered_records_all]
    ordered_sample_ids_target = [str(row["sample_id"]) for row in ordered_records]
    ordered_hash_all = _sha256_lines(ordered_sample_ids_all)
    ordered_hash_target = _sha256_lines(ordered_sample_ids_target)

    with _resume_lock(resume_paths, stale_seconds=feature_cfg["lock_stale_seconds"]):
        _append_resume_event(
            resume_paths,
            event_type="run_start",
            details={
                "mode": "extract.intphys2",
                "signature": paths.signature,
                "target_max_samples": int(feature_cfg["max_samples"]),
                "target_samples": len(ordered_records),
                "ordered_samples_all": len(ordered_records_all),
                "config_hash": config_hash,
                "run_started_utc": run_started_utc,
            },
        )

        if _is_valid_cache(paths) and not feature_cfg["force_reextract"]:
            manifest = _load_json(paths.manifest_path)
            if _cache_satisfies_target(manifest, requested_samples=len(ordered_records), feature_cfg=feature_cfg):
                _append_resume_event(
                    resume_paths,
                    event_type="run_complete",
                    details={
                        "resumed": False,
                        "skipped": True,
                        "reason": "valid_cache_exists",
                        "n_samples": int(manifest.get("features", {}).get("n_samples", 0)),
                    },
                )
                print(
                    "[extract.intphys2] valid cache already satisfies request; "
                    f"signature={paths.signature} n_samples={manifest.get('features', {}).get('n_samples', 0)}"
                )
                elapsed_seconds = max(0.0, time.perf_counter() - wall_start)
                return {
                    "feature_cache_dir": str(paths.cache_dir),
                    "signature": paths.signature,
                    "skipped": True,
                    "reason": "valid_cache_exists",
                    "elapsed_seconds": elapsed_seconds,
                    "seconds_per_processed_sample": 0.0,
                    "warning_log": str(warning_log_path),
                }

        state = None if not feature_cfg["resume_enabled"] else _load_resume_state(resume_paths.state_path)
        if state is None:
            state = _new_resume_state(
                signature=paths.signature,
                config_hash=config_hash,
                ordered_hash_all=ordered_hash_all,
                ordered_count_all=len(ordered_records_all),
                target_max_samples=int(feature_cfg["max_samples"]),
                target_samples=len(ordered_records),
                chunk_size=int(feature_cfg["chunk_size"]),
            )
            _write_json_atomic(resume_paths.state_path, state)
            _append_resume_event(
                resume_paths,
                event_type="resume_not_possible",
                details={
                    "reason": "resume_disabled" if not feature_cfg["resume_enabled"] else "no_existing_state",
                    "resume_from_offset": 0,
                },
            )
        else:
            state = _validate_or_reset_resume_state(
                state=state,
                signature=paths.signature,
                config_hash=config_hash,
                ordered_hash_all=ordered_hash_all,
                ordered_count_all=len(ordered_records_all),
                target_max_samples=int(feature_cfg["max_samples"]),
                target_samples=len(ordered_records),
                feature_cfg=feature_cfg,
                paths=paths,
                resume_paths=resume_paths,
            )

        start_offset = int(state.get("n_completed_samples", 0))
        if start_offset > len(ordered_records):
            start_offset = len(ordered_records)

        _append_resume_event(
            resume_paths,
            event_type="resume_plan",
            details={
                "resumed": bool(start_offset > 0),
                "resume_from_sample_offset": start_offset,
                "target_samples": len(ordered_records),
                "source_state_path": str(resume_paths.state_path),
            },
        )
        print(
            "[extract.intphys2] resume status: "
            f"resumed={start_offset > 0} offset={start_offset} target={len(ordered_records)} "
            f"signature={paths.signature}"
        )

        selected_layers = tuple(int(v) for v in state.get("selected_layers", []))
        adapter_metadata = state.get("adapter_metadata", {})

        while start_offset < len(ordered_records):
            chunk_size = int(feature_cfg["chunk_size"])
            end_offset = min(start_offset + chunk_size, len(ordered_records))
            chunk_records = ordered_records[start_offset:end_offset]
            chunk_id = int(state.get("next_chunk_id", 0))

            _append_resume_event(
                resume_paths,
                event_type="chunk_start",
                details={
                    "chunk_id": chunk_id,
                    "sample_offset_start": start_offset,
                    "sample_offset_end": end_offset,
                    "n_samples": len(chunk_records),
                },
            )

            chunk_rows, chunk_pooled, chunk_tokens, chunk_layers, chunk_metadata = _extract_chunk_records(
                records=chunk_records,
                feature_index_offset=start_offset,
                adapter=adapter,
                requested_layer_ids=requested_layer_ids,
                decode_cfg=decode_cfg,
                include_pooled=bool(feature_cfg["include_pooled"]),
                include_tokens=bool(feature_cfg["include_tokens"]),
            )

            if not selected_layers:
                selected_layers = tuple(chunk_layers)
                adapter_metadata = dict(chunk_metadata)
            elif tuple(chunk_layers) != tuple(selected_layers):
                raise FeatureCacheError(
                    "Adapter returned inconsistent selected_layers across chunks: "
                    f"state={selected_layers}, chunk={chunk_layers}"
                )

            chunk_dir = resume_paths.chunks_dir / f"{chunk_id:06d}_{start_offset:09d}_{end_offset:09d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_index_path = chunk_dir / "index.parquet"
            chunk_pooled_path = chunk_dir / "features_pooled.pt"
            chunk_tokens_path = chunk_dir / "features_tokens.pt"

            _write_index(chunk_index_path, chunk_rows)
            if feature_cfg["include_pooled"]:
                _torch_save_atomic(
                    chunk_pooled_path,
                    {"selected_layers": list(selected_layers), "by_layer": chunk_pooled},
                )
            if feature_cfg["include_tokens"]:
                _torch_save_atomic(
                    chunk_tokens_path,
                    {"selected_layers": list(selected_layers), "by_layer": chunk_tokens},
                )

            state_chunks = list(state.get("completed_chunks", []))
            state_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "n_samples": len(chunk_rows),
                    "index_path": str(chunk_index_path),
                    "pooled_path": str(chunk_pooled_path) if feature_cfg["include_pooled"] else "",
                    "tokens_path": str(chunk_tokens_path) if feature_cfg["include_tokens"] else "",
                }
            )

            state.update(
                {
                    "status": "in_progress",
                    "target_max_samples": int(feature_cfg["max_samples"]),
                    "total_target_samples": len(ordered_records),
                    "completed_chunks": state_chunks,
                    "n_completed_samples": end_offset,
                    "resume_from_sample_offset": end_offset,
                    "resume_from_chunk_id": chunk_id + 1,
                    "next_chunk_id": chunk_id + 1,
                    "selected_layers": list(selected_layers),
                    "adapter_metadata": adapter_metadata,
                    "last_writer_job_id": _slurm_job_id(),
                    "hostname": socket.gethostname(),
                    "updated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
            _write_json_atomic(resume_paths.state_path, state)
            _append_resume_event(
                resume_paths,
                event_type="chunk_complete",
                details={
                    "chunk_id": chunk_id,
                    "sample_offset_start": start_offset,
                    "sample_offset_end": end_offset,
                    "n_samples": len(chunk_rows),
                },
            )

            start_offset = end_offset

        _append_resume_event(resume_paths, event_type="finalize_start", details={})
        index_rows, pooled_by_layer, tokens_by_layer = _materialize_from_chunks(
            state=state,
            include_pooled=bool(feature_cfg["include_pooled"]),
            include_tokens=bool(feature_cfg["include_tokens"]),
            selected_layers=tuple(selected_layers),
            target_samples=len(ordered_records),
        )

        if len(index_rows) != len(ordered_records):
            raise FeatureCacheError(
                "Resume materialization row mismatch: "
                f"expected={len(ordered_records)} got={len(index_rows)}"
            )

        _write_index(paths.index_path, index_rows)
        if feature_cfg["include_pooled"]:
            _torch_save_atomic(
                paths.pooled_path,
                {"selected_layers": list(selected_layers), "by_layer": pooled_by_layer},
            )
        if feature_cfg["include_tokens"]:
            _torch_save_atomic(
                paths.tokens_path,
                {"selected_layers": list(selected_layers), "by_layer": tokens_by_layer},
            )

        raw_reused_samples = int(state.get("n_completed_samples_initial", 0))
        reused_samples = min(max(raw_reused_samples, 0), len(index_rows))
        resume_start_offset = reused_samples
        new_samples = len(index_rows) - reused_samples
        run_finished_utc = datetime.now(tz=timezone.utc).isoformat()
        elapsed_seconds = max(0.0, time.perf_counter() - wall_start)
        seconds_per_processed_sample = elapsed_seconds / float(new_samples) if new_samples > 0 else 0.0
        seconds_per_total_sample = elapsed_seconds / float(len(index_rows)) if index_rows else 0.0
        aggregated_elapsed_seconds = float(state.get("aggregated_elapsed_seconds", 0.0)) + elapsed_seconds
        aggregated_processed_samples = int(state.get("aggregated_processed_samples", 0)) + int(new_samples)
        aggregated_seconds_per_processed_sample = (
            aggregated_elapsed_seconds / float(aggregated_processed_samples)
            if aggregated_processed_samples > 0
            else 0.0
        )
        aggregated_seconds_per_total_sample = (
            aggregated_elapsed_seconds / float(len(index_rows))
            if index_rows
            else 0.0
        )
        manifest = _build_manifest(
            paths=paths,
            split_dir=split_dir,
            split_manifest_path=split_manifest_path,
            split_manifest=split_manifest,
            split_names=split_names,
            metadata_file=metadata_file,
            metadata_sha=metadata_sha,
            backbone_name=backbone_name,
            backbone_kwargs=backbone_kwargs,
            adapter_metadata=adapter_metadata,
            decode_cfg=decode_cfg,
            feature_cfg=feature_cfg,
            selected_layers=tuple(selected_layers),
            index_rows=index_rows,
            resume_paths=resume_paths,
            resumed=bool(reused_samples > 0),
            reused_samples=reused_samples,
            new_samples=new_samples,
            resume_start_offset=resume_start_offset,
            ordered_hash_all=ordered_hash_all,
            ordered_hash_target=ordered_hash_target,
            run_started_utc=run_started_utc,
            run_finished_utc=run_finished_utc,
            elapsed_seconds=elapsed_seconds,
            seconds_per_processed_sample=seconds_per_processed_sample,
            seconds_per_total_sample=seconds_per_total_sample,
            aggregated_elapsed_seconds=aggregated_elapsed_seconds,
            aggregated_processed_samples=aggregated_processed_samples,
            aggregated_seconds_per_processed_sample=aggregated_seconds_per_processed_sample,
            aggregated_seconds_per_total_sample=aggregated_seconds_per_total_sample,
        )
        _write_json_atomic(paths.manifest_path, manifest)

        state.update(
            {
                "status": "complete",
                "n_completed_samples": len(index_rows),
                "resume_from_sample_offset": len(index_rows),
                "resume_from_chunk_id": int(state.get("next_chunk_id", 0)),
                "total_target_samples": len(index_rows),
                "last_writer_job_id": _slurm_job_id(),
                "hostname": socket.gethostname(),
                "updated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "aggregated_elapsed_seconds": aggregated_elapsed_seconds,
                "aggregated_processed_samples": aggregated_processed_samples,
                "completed_runs": int(state.get("completed_runs", 0)) + 1,
            }
        )
        _write_json_atomic(resume_paths.state_path, state)
        _append_resume_event(
            resume_paths,
            event_type="finalize_complete",
            details={"n_samples": len(index_rows)},
        )
        _append_resume_event(
            resume_paths,
            event_type="run_complete",
            details={
                "resumed": bool(reused_samples > 0),
                "reused_samples": reused_samples,
                "new_samples": new_samples,
                "resume_from_sample_offset": resume_start_offset,
                "n_samples": len(index_rows),
                "elapsed_seconds": elapsed_seconds,
                "seconds_per_processed_sample": seconds_per_processed_sample,
                "aggregated_elapsed_seconds": aggregated_elapsed_seconds,
                "aggregated_processed_samples": aggregated_processed_samples,
            },
        )

    print(
        "[extract.intphys2] completed: "
        f"signature={paths.signature} resumed={reused_samples > 0} "
        f"reused_samples={reused_samples} new_samples={new_samples} total={len(index_rows)} "
        f"elapsed_s={elapsed_seconds:.3f} sec_per_processed_sample={seconds_per_processed_sample:.6f} "
        f"elapsed_agg_s={aggregated_elapsed_seconds:.3f}"
    )
    return {
        "feature_cache_dir": str(paths.cache_dir),
        "signature": paths.signature,
        "skipped": False,
        "n_samples": len(index_rows),
        "selected_layers": list(selected_layers),
        "warning_log": str(warning_log_path),
        "resumed": bool(reused_samples > 0),
        "resume_from_sample_offset": resume_start_offset,
        "resume_events": str(resume_paths.events_path),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_processed_sample": seconds_per_processed_sample,
        "seconds_per_total_sample": seconds_per_total_sample,
        "aggregated_elapsed_seconds": aggregated_elapsed_seconds,
        "aggregated_processed_samples": aggregated_processed_samples,
        "aggregated_seconds_per_processed_sample": aggregated_seconds_per_processed_sample,
        "aggregated_seconds_per_total_sample": aggregated_seconds_per_total_sample,
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
    exact_paths = FeatureCachePaths(
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        index_path=cache_dir / "index.parquet",
        pooled_path=cache_dir / "features_pooled.pt",
        tokens_path=cache_dir / "features_tokens.pt",
        signature=signature,
    )
    if exact_paths.manifest_path.exists():
        return exact_paths

    compatible = _find_compatible_feature_cache(
        base_dir=base_dir,
        backbone_id=backbone_id,
        split_key=split_key,
        split_manifest_metadata_sha256=str(split_manifest.get("metadata_sha256", "")),
        feature_cfg=feature_cfg,
        backbone_name=backbone_name,
        backbone_metadata=backbone_metadata,
        decode_cfg=payload["decode"],
    )
    return compatible or exact_paths


def _find_compatible_feature_cache(
    *,
    base_dir: Path,
    backbone_id: str,
    split_key: str,
    split_manifest_metadata_sha256: str,
    feature_cfg: dict[str, Any],
    backbone_name: str,
    backbone_metadata: dict[str, Any],
    decode_cfg: dict[str, Any],
) -> FeatureCachePaths | None:
    root = base_dir / backbone_id / split_key
    if not root.exists():
        return None

    expected_layers = (
        [int(v) for v in feature_cfg["layer_ids"]]
        if feature_cfg["layer_ids"]
        else [int(v) for v in backbone_metadata.get("selected_layers", [])]
    )
    expected_variant = str(backbone_metadata.get("variant", "")).strip()

    matches: list[tuple[float, FeatureCachePaths]] = []
    for manifest_path in root.glob("*/manifest.json"):
        cache_dir = manifest_path.parent
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            continue
        if not _manifest_matches_requested_cache(
            manifest,
            split_manifest_metadata_sha256=split_manifest_metadata_sha256,
            feature_cfg=feature_cfg,
            backbone_name=backbone_name,
            expected_variant=expected_variant,
            expected_layers=expected_layers,
            decode_cfg=decode_cfg,
        ):
            continue
        matches.append((manifest_path.stat().st_mtime, _feature_cache_paths(cache_dir)))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _manifest_matches_requested_cache(
    manifest: dict[str, Any],
    *,
    split_manifest_metadata_sha256: str,
    feature_cfg: dict[str, Any],
    backbone_name: str,
    expected_variant: str,
    expected_layers: list[int],
    decode_cfg: dict[str, Any],
) -> bool:
    if str(manifest.get("kind", "")) != "intphys2_feature_cache":
        return False

    split = manifest.get("split", {})
    if str(split.get("metadata_sha256", "")) != split_manifest_metadata_sha256:
        return False
    if [str(item) for item in split.get("names", [])] != [str(item) for item in feature_cfg["split_names"]]:
        return False

    backbone = manifest.get("backbone", {})
    if str(backbone.get("name", "")) != backbone_name:
        return False

    metadata = backbone.get("metadata", {})
    if str(metadata.get("variant", "")).strip() != expected_variant:
        return False

    features = manifest.get("features", {})
    if bool(features.get("include_pooled", False)) != bool(feature_cfg["include_pooled"]):
        return False
    if bool(features.get("include_tokens", False)) != bool(feature_cfg["include_tokens"]):
        return False
    if [int(v) for v in features.get("selected_layers", [])] != expected_layers:
        return False

    manifest_decode = manifest.get("decode", {})
    return {
        "num_frames": int(manifest_decode.get("num_frames", -1)),
        "sampling": str(manifest_decode.get("sampling", "")),
        "crop_size": int(manifest_decode.get("crop_size", -1)),
    } == {
        "num_frames": int(decode_cfg["num_frames"]),
        "sampling": str(decode_cfg["sampling"]),
        "crop_size": int(decode_cfg["crop_size"]),
    }


def _feature_cache_paths(cache_dir: Path) -> FeatureCachePaths:
    return FeatureCachePaths(
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        index_path=cache_dir / "index.parquet",
        pooled_path=cache_dir / "features_pooled.pt",
        tokens_path=cache_dir / "features_tokens.pt",
        signature=cache_dir.name,
    )


def has_valid_feature_cache(config: dict[str, Any]) -> bool:
    try:
        paths = resolve_expected_feature_cache_paths(config)
    except Exception:
        return False
    if not _is_valid_cache(paths):
        return False
    try:
        manifest = _load_json(paths.manifest_path)
        feature_cfg = _feature_cfg(config)
        return _cache_satisfies_target(
            manifest,
            requested_samples=int(manifest.get("features", {}).get("n_samples", 0))
            if int(feature_cfg.get("max_samples", 0)) <= 0
            else int(feature_cfg["max_samples"]),
            feature_cfg=feature_cfg,
        )
    except Exception:
        return False


def load_feature_cache_for_config(
    config: dict[str, Any],
    *,
    feature_view: str | None = None,
) -> dict[str, Any]:
    paths = resolve_expected_feature_cache_paths(config)
    feature_cfg = _feature_cfg(config)
    if not _is_valid_cache(paths):
        raise FeatureCacheError(
            "Feature cache is missing or invalid for current config. "
            "Run `python run.py extract.intphys2` first."
        )

    manifest = _load_json(paths.manifest_path)
    requested_samples = int(feature_cfg["max_samples"])
    if requested_samples <= 0:
        requested_samples = int(manifest.get("features", {}).get("n_samples", 0))
    if not _cache_satisfies_target(manifest, requested_samples=requested_samples, feature_cfg=feature_cfg):
        raise FeatureCacheError(
            "Feature cache exists but does not satisfy requested sample count. "
            "Re-run `python run.py extract.intphys2`."
        )
    frame = _load_index(paths.index_path)

    pooled_payload: dict[str, Any] | None = None
    tokens_payload: dict[str, Any] | None = None

    view = str(feature_view or "").strip().lower()
    load_pooled = view in {"", "pooled"}
    load_tokens = view in {"", "tokens_mean", "tokens"}

    if load_pooled and str(manifest.get("files", {}).get("pooled", "")):
        pooled_payload = torch.load(str(paths.pooled_path), map_location="cpu", weights_only=False)
    if load_tokens and str(manifest.get("files", {}).get("tokens", "")):
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

    max_samples_raw = raw.get("max_samples", 0)
    try:
        max_samples = int(max_samples_raw)
    except (TypeError, ValueError) as exc:
        raise FeatureConfigError("feature_cache.max_samples must be an integer") from exc

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
        "max_samples": max_samples,
        "force_reextract": bool(raw.get("force_reextract", False)),
        "resume_enabled": bool(raw.get("resume_enabled", True)),
        "resume_strict": bool(raw.get("resume_strict", True)),
        "chunk_size": max(1, int(raw.get("chunk_size", 64))),
        "lock_stale_seconds": max(60, int(raw.get("lock_stale_seconds", 86400))),
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


def _limit_ordered_records(
    ordered_records: list[dict[str, Any]],
    max_samples: int,
) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return ordered_records
    return ordered_records[:max_samples]


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


def _resume_paths(cache_dir: Path) -> ResumePaths:
    root_dir = cache_dir / ".resume"
    return ResumePaths(
        root_dir=root_dir,
        chunks_dir=root_dir / "chunks",
        state_path=root_dir / "state.json",
        events_path=root_dir / "events.jsonl",
        lock_path=root_dir / "lock",
    )


def _slurm_job_id() -> str:
    return str(os.environ.get("SLURM_JOB_ID", ""))


def _resume_config_payload(
    *,
    config: dict[str, Any],
    feature_cfg: dict[str, Any],
    split_manifest: dict[str, Any],
    split_dir: Path,
    metadata_file: str,
    backbone_name: str,
    backbone_kwargs: dict[str, Any],
    decode_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "extract.intphys2",
        "metadata_file": str(metadata_file),
        "split_dir": str(split_dir),
        "split_manifest_metadata_sha256": str(split_manifest.get("metadata_sha256", "")),
        "split_names": list(feature_cfg["split_names"]),
        "layer_ids": list(feature_cfg["layer_ids"]),
        "include_pooled": bool(feature_cfg["include_pooled"]),
        "include_tokens": bool(feature_cfg["include_tokens"]),
        "backbone_name": str(backbone_name),
        "backbone_kwargs": dict(backbone_kwargs),
        "decode": dict(decode_cfg),
        "videos_root": str(config.get("videos_root", "")),
    }


def _new_resume_state(
    *,
    signature: str,
    config_hash: str,
    ordered_hash_all: str,
    ordered_count_all: int,
    target_max_samples: int,
    target_samples: int,
    chunk_size: int,
) -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc).isoformat()
    return {
        "version": 1,
        "status": "in_progress",
        "signature": str(signature),
        "config_hash": str(config_hash),
        "ordered_sample_ids_hash": str(ordered_hash_all),
        "ordered_sample_count": int(ordered_count_all),
        "chunk_size": int(chunk_size),
        "target_max_samples": int(target_max_samples),
        "total_target_samples": int(target_samples),
        "completed_chunks": [],
        "n_completed_samples": 0,
        "n_completed_samples_initial": 0,
        "resume_from_sample_offset": 0,
        "resume_from_chunk_id": 0,
        "next_chunk_id": 0,
        "selected_layers": [],
        "adapter_metadata": {},
        "created_at_utc": now,
        "updated_at_utc": now,
        "last_writer_job_id": _slurm_job_id(),
        "hostname": socket.gethostname(),
        "aggregated_elapsed_seconds": 0.0,
        "aggregated_processed_samples": 0,
        "completed_runs": 0,
    }


def _load_resume_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    return payload


def _validate_or_reset_resume_state(
    *,
    state: dict[str, Any],
    signature: str,
    config_hash: str,
    ordered_hash_all: str,
    ordered_count_all: int,
    target_max_samples: int,
    target_samples: int,
    feature_cfg: dict[str, Any],
    paths: FeatureCachePaths,
    resume_paths: ResumePaths,
) -> dict[str, Any]:
    state_signature = str(state.get("signature", ""))
    state_config_hash = str(state.get("config_hash", ""))
    state_ordered_hash = str(state.get("ordered_sample_ids_hash", ""))
    state_ordered_count = int(state.get("ordered_sample_count", -1))

    mismatch = (
        state_signature != str(signature)
        or state_config_hash != str(config_hash)
        or state_ordered_hash != str(ordered_hash_all)
        or state_ordered_count != int(ordered_count_all)
    )
    if mismatch:
        if feature_cfg["resume_strict"]:
            raise FeatureCacheError(
                "Resume state does not match current extraction configuration. "
                "Set feature_cache.resume_strict=false to reset state automatically."
            )
        _append_resume_event(
            resume_paths,
            event_type="resume_not_possible",
            details={"reason": "state_mismatch_reset"},
        )
        _reset_resume_state(paths, resume_paths)
        return _new_resume_state(
            signature=signature,
            config_hash=config_hash,
            ordered_hash_all=ordered_hash_all,
            ordered_count_all=ordered_count_all,
            target_max_samples=target_max_samples,
            target_samples=target_samples,
            chunk_size=int(feature_cfg["chunk_size"]),
        )

    n_completed = int(state.get("n_completed_samples", 0))
    if n_completed < 0 or n_completed > int(ordered_count_all):
        raise FeatureCacheError(
            "Resume state has invalid n_completed_samples "
            f"({n_completed}) for ordered_count={ordered_count_all}."
        )

    state["target_max_samples"] = int(target_max_samples)
    state["total_target_samples"] = int(target_samples)
    state["n_completed_samples_initial"] = int(n_completed)
    state["updated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    state["last_writer_job_id"] = _slurm_job_id()
    state["hostname"] = socket.gethostname()
    state["aggregated_elapsed_seconds"] = float(state.get("aggregated_elapsed_seconds", 0.0))
    state["aggregated_processed_samples"] = int(state.get("aggregated_processed_samples", 0))
    state["completed_runs"] = int(state.get("completed_runs", 0))
    return state


def _cache_satisfies_target(
    manifest: dict[str, Any],
    *,
    requested_samples: int,
    feature_cfg: dict[str, Any],
) -> bool:
    features = manifest.get("features", {})
    n_samples = int(features.get("n_samples", 0))
    if n_samples <= 0:
        return False
    if requested_samples > 0 and n_samples < requested_samples:
        return False

    requested_max_samples = int(feature_cfg.get("max_samples", 0))
    cached_requested = int(features.get("requested_max_samples", features.get("max_samples", 0)))
    if requested_max_samples <= 0 and cached_requested > 0:
        return False
    return True


def _reset_resume_state(paths: FeatureCachePaths, resume_paths: ResumePaths) -> None:
    if resume_paths.root_dir.exists():
        shutil.rmtree(resume_paths.root_dir)
    for file_path in [paths.manifest_path, paths.index_path, paths.pooled_path, paths.tokens_path]:
        if file_path.exists():
            file_path.unlink()


@contextmanager
def _resume_lock(resume_paths: ResumePaths, *, stale_seconds: int) -> Any:
    resume_paths.root_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    now_utc = datetime.now(tz=timezone.utc).isoformat()
    payload = {
        "token": token,
        "created_at_utc": now_utc,
        "hostname": socket.gethostname(),
        "job_id": _slurm_job_id(),
        "pid": os.getpid(),
    }

    for _ in range(2):
        try:
            fd = os.open(str(resume_paths.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
            break
        except FileExistsError:
            stale = _is_stale_lock(resume_paths.lock_path, stale_seconds=stale_seconds)
            if not stale:
                raise FeatureCacheError(
                    "Another extraction process holds the resume lock: "
                    f"{resume_paths.lock_path}"
                )
            resume_paths.lock_path.unlink(missing_ok=True)
    else:
        raise FeatureCacheError(f"Failed to acquire resume lock: {resume_paths.lock_path}")

    try:
        yield
    finally:
        if resume_paths.lock_path.exists():
            try:
                existing = json.loads(resume_paths.lock_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            if str(existing.get("token", "")) == token:
                resume_paths.lock_path.unlink(missing_ok=True)


def _is_stale_lock(lock_path: Path, *, stale_seconds: int) -> bool:
    if not lock_path.exists():
        return True
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        created = str(payload.get("created_at_utc", ""))
        if not created:
            return True
        created_ts = datetime.fromisoformat(created)
        age = (datetime.now(tz=timezone.utc) - created_ts).total_seconds()
        return age > float(stale_seconds)
    except Exception:
        return True


def _append_resume_event(
    resume_paths: ResumePaths,
    *,
    event_type: str,
    details: dict[str, Any],
) -> None:
    resume_paths.root_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": str(event_type),
        "job_id": _slurm_job_id(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "details": details,
    }
    with resume_paths.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _torch_save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _extract_chunk_records(
    *,
    records: list[dict[str, Any]],
    feature_index_offset: int,
    adapter: Any,
    requested_layer_ids: list[int] | None,
    decode_cfg: dict[str, Any],
    include_pooled: bool,
    include_tokens: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    tuple[int, ...],
    dict[str, Any],
]:
    chunk_rows: list[dict[str, Any]] = []
    pooled_acc: dict[int, list[torch.Tensor]] = {}
    tokens_acc: dict[int, list[torch.Tensor]] = {}
    selected_layers: tuple[int, ...] | None = None
    adapter_metadata: dict[str, Any] = {}

    for local_index, record in enumerate(records):
        clip = _decode_video_clip(
            video_path=record["video_path"],
            num_frames=int(decode_cfg["num_frames"]),
            crop_size=int(decode_cfg["crop_size"]),
        )
        with torch.no_grad():
            features = adapter.extract(clip, layer_ids=requested_layer_ids)

        if selected_layers is None:
            selected_layers = tuple(int(v) for v in features.selected_layers)
            adapter_metadata = dict(features.metadata)
        elif tuple(int(v) for v in features.selected_layers) != tuple(selected_layers):
            raise FeatureCacheError(
                "Adapter returned inconsistent selected_layers within chunk: "
                f"first={selected_layers}, current={features.selected_layers}"
            )

        for layer in features.selected_layers:
            layer_id = int(layer)
            if include_pooled:
                pooled_acc.setdefault(layer_id, []).append(features.pooled_by_layer[layer].detach().cpu())
            if include_tokens:
                tokens_acc.setdefault(layer_id, []).append(features.tokens_by_layer[layer].detach().cpu())

        chunk_rows.append(
            {
                "feature_index": int(feature_index_offset + local_index),
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
        raise FeatureCacheError("Chunk extraction produced no outputs.")

    pooled_by_layer = {
        layer: torch.cat(values, dim=0)
        for layer, values in sorted(pooled_acc.items())
    } if include_pooled else {}
    tokens_by_layer = {
        layer: torch.cat(values, dim=0)
        for layer, values in sorted(tokens_acc.items())
    } if include_tokens else {}

    return chunk_rows, pooled_by_layer, tokens_by_layer, tuple(selected_layers), adapter_metadata


def _materialize_from_chunks(
    *,
    state: dict[str, Any],
    include_pooled: bool,
    include_tokens: bool,
    selected_layers: tuple[int, ...],
    target_samples: int,
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    completed_chunks = sorted(
        list(state.get("completed_chunks", [])),
        key=lambda chunk: int(chunk.get("start_offset", 0)),
    )

    index_rows: list[dict[str, Any]] = []
    pooled_acc: dict[int, list[torch.Tensor]] = {}
    tokens_acc: dict[int, list[torch.Tensor]] = {}

    for chunk in completed_chunks:
        if len(index_rows) >= target_samples:
            break
        chunk_index_path = Path(str(chunk.get("index_path", "")))
        if not chunk_index_path.exists():
            raise FeatureCacheError(f"Missing chunk index file: {chunk_index_path}")
        chunk_rows = _load_index(chunk_index_path).to_dict(orient="records")
        n_take = min(len(chunk_rows), max(0, target_samples - len(index_rows)))
        chunk_rows = chunk_rows[:n_take]
        index_rows.extend(chunk_rows)

        if include_pooled:
            chunk_pooled_path = Path(str(chunk.get("pooled_path", "")))
            if not chunk_pooled_path.exists():
                raise FeatureCacheError(f"Missing chunk pooled payload: {chunk_pooled_path}")
            payload = torch.load(str(chunk_pooled_path), map_location="cpu", weights_only=False)
            for layer in selected_layers:
                pooled_acc.setdefault(int(layer), []).append(payload["by_layer"][int(layer)][:n_take])

        if include_tokens:
            chunk_tokens_path = Path(str(chunk.get("tokens_path", "")))
            if not chunk_tokens_path.exists():
                raise FeatureCacheError(f"Missing chunk tokens payload: {chunk_tokens_path}")
            payload = torch.load(str(chunk_tokens_path), map_location="cpu", weights_only=False)
            for layer in selected_layers:
                tokens_acc.setdefault(int(layer), []).append(payload["by_layer"][int(layer)][:n_take])

    pooled_by_layer = {
        layer: torch.cat(values, dim=0)
        for layer, values in sorted(pooled_acc.items())
    } if include_pooled else {}
    tokens_by_layer = {
        layer: torch.cat(values, dim=0)
        for layer, values in sorted(tokens_acc.items())
    } if include_tokens else {}
    return index_rows, pooled_by_layer, tokens_by_layer


def _build_manifest(
    *,
    paths: FeatureCachePaths,
    split_dir: Path,
    split_manifest_path: Path,
    split_manifest: dict[str, Any],
    split_names: list[str],
    metadata_file: Path,
    metadata_sha: str,
    backbone_name: str,
    backbone_kwargs: dict[str, Any],
    adapter_metadata: dict[str, Any],
    decode_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    selected_layers: tuple[int, ...],
    index_rows: list[dict[str, Any]],
    resume_paths: ResumePaths,
    resumed: bool,
    reused_samples: int,
    new_samples: int,
    resume_start_offset: int,
    ordered_hash_all: str,
    ordered_hash_target: str,
    run_started_utc: str,
    run_finished_utc: str,
    elapsed_seconds: float,
    seconds_per_processed_sample: float,
    seconds_per_total_sample: float,
    aggregated_elapsed_seconds: float,
    aggregated_processed_samples: int,
    aggregated_seconds_per_processed_sample: float,
    aggregated_seconds_per_total_sample: float,
) -> dict[str, Any]:
    return {
        "version": 2,
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
            "max_samples": int(feature_cfg["max_samples"]),
            "requested_max_samples": int(feature_cfg["max_samples"]),
            "selected_layers": list(selected_layers),
            "n_samples": len(index_rows),
            "sample_ids_sha256": _sha256_lines([row["sample_id"] for row in index_rows]),
            "ordered_sample_ids_sha256_all": ordered_hash_all,
            "ordered_sample_ids_sha256_target": ordered_hash_target,
        },
        "resume": {
            "enabled": bool(feature_cfg["resume_enabled"]),
            "resumed": bool(resumed),
            "source_state_path": str(resume_paths.state_path),
            "events_path": str(resume_paths.events_path),
            "resume_start_offset": int(resume_start_offset),
            "reused_samples": int(reused_samples),
            "new_samples": int(new_samples),
        },
        "timing": {
            "run_started_utc": str(run_started_utc),
            "run_finished_utc": str(run_finished_utc),
            "elapsed_seconds": float(elapsed_seconds),
            "processed_samples": int(new_samples),
            "seconds_per_processed_sample": float(seconds_per_processed_sample),
            "seconds_per_total_sample": float(seconds_per_total_sample),
            "aggregated": {
                "elapsed_seconds": float(aggregated_elapsed_seconds),
                "processed_samples": int(aggregated_processed_samples),
                "seconds_per_processed_sample": float(aggregated_seconds_per_processed_sample),
                "seconds_per_total_sample": float(aggregated_seconds_per_total_sample),
            },
        },
        "files": {
            "index": paths.index_path.name,
            "pooled": paths.pooled_path.name if feature_cfg["include_pooled"] else "",
            "tokens": paths.tokens_path.name if feature_cfg["include_tokens"] else "",
        },
    }


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
