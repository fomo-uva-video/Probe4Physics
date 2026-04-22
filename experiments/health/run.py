from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmarks.intphys2.data import load_intphys2_rows, normalize_intphys2_row
from benchmarks.mvp.data import load_mvp_rows
from benchmarks.mvp.selection import derive_sample_id, derive_video_ref
from benchmarks.ssv2.data import load_ssv2_annotations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

BACKBONE_ORDER = [
    "jepa_v1",
    "jepa_v2",
    "jepa_v2_1",
    "videomae",
    "videomae_v2",
    "ltx_video",
]

# Largest variants requested for health checks.
LARGEST_VARIANTS = {
    "jepa_v1": "vith16_384",
    "jepa_v2": "vitg_384",
    "jepa_v2_1": "vitG_384",
    "videomae": "vit_huge_16_224",
    "videomae_v2": "vit_giant_16_224",
    "ltx_video": "ltx_video_main",
}


def run_health(config: dict[str, Any]) -> dict[str, Any]:
    _ = config
    backbone_cfg = _load_yaml(CONFIGS_DIR / "backbones.yaml")
    mvp_cfg = _load_yaml(CONFIGS_DIR / "mvp.yaml")
    intphys2_cfg = _load_yaml(CONFIGS_DIR / "intphys2.yaml")
    ssv2_cfg = _load_yaml(CONFIGS_DIR / "ssv2.yaml")

    backbones = [
        _check_backbone(name, backbone_cfg)
        for name in BACKBONE_ORDER
    ]
    datasets = [
        _check_mvp(mvp_cfg),
        _check_intphys2(intphys2_cfg),
        _check_ssv2(ssv2_cfg),
    ]

    checks: list[dict[str, Any]] = []
    for item in backbones + datasets:
        checks.extend(item.get("checks", []))

    total = len(checks)
    passed = sum(1 for check in checks if check.get("status") == "pass")
    failed = sum(1 for check in checks if check.get("status") == "fail")
    ok = failed == 0

    report = {
        "ok": ok,
        "exit_code": 0 if ok else 1,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
        },
        "tested_backbones": [
            {
                "name": item.get("name"),
                "variant": item.get("variant"),
                "status": item.get("status"),
            }
            for item in backbones
        ],
        "tested_datasets": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
            }
            for item in datasets
        ],
        "backbones": backbones,
        "datasets": datasets,
    }
    report["human_report"] = _format_human_report(report)
    return report


def _check_backbone(name: str, backbone_cfg: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    section = backbone_cfg.get(name)
    variant = LARGEST_VARIANTS.get(name, "")
    if not isinstance(section, dict):
        checks.append(_check_result("config_section_exists", False, f"Missing '{name}' in backbones.yaml"))
        return _bundle("backbone", name, variant, checks)

    variants = section.get("variants", {})
    if not isinstance(variants, dict) or variant not in variants:
        checks.append(
            _check_result(
                "variant_wired",
                False,
                f"Variant '{variant}' not found under '{name}.variants'.",
            )
        )
        return _bundle("backbone", name, variant, checks)

    checks.append(_check_result("variant_wired", True, f"Configured variant '{variant}' found."))
    variant_cfg = variants[variant]

    resolved_checkpoint: Path | None = None
    if "checkpoints_dir" in section and isinstance(variant_cfg, dict) and "checkpoint_filename" in variant_cfg:
        checkpoints_dir = _as_project_path(str(section.get("checkpoints_dir", "")))
        checkpoint_filename = str(variant_cfg.get("checkpoint_filename", "")).strip()
        resolved_checkpoint = checkpoints_dir / checkpoint_filename
        checks.append(
            _check_result(
                "checkpoint_present",
                resolved_checkpoint.exists(),
                str(resolved_checkpoint),
            )
        )
    else:
        checks.append(_check_result("checkpoint_present", True, "Not required (HF-backed backbone)."))

    probe_kwargs: dict[str, Any] = {"variant": variant, "device": "cpu"}
    if resolved_checkpoint is not None:
        probe_kwargs["checkpoint_path"] = str(resolved_checkpoint)

    probe_ok, probe_detail = _run_backbone_probe_subprocess(name, probe_kwargs)
    checks.append(_check_result("synthetic_forward", probe_ok, probe_detail))
    return _bundle("backbone", name, variant, checks)


def _run_backbone_probe_subprocess(name: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
    script = f"""
import json
import traceback
import torch
from models import create_adapter

name = {name!r}
kwargs = {json.dumps(kwargs)}
try:
    adapter = create_adapter(name, **kwargs)
    frames = int(getattr(adapter, "frames_per_clip", 16))
    crop = int(getattr(adapter, "crop_size", 224))
    clips = torch.zeros((1, 3, frames, crop, crop), dtype=torch.float32)
    selected = getattr(adapter, "selected_layers", ())
    layer_ids = [int(selected[0])] if isinstance(selected, (list, tuple)) and selected else None
    with torch.no_grad():
        features = adapter.extract(clips, layer_ids=layer_ids)
    payload = {{
        "ok": True,
        "frames_per_clip": frames,
        "crop_size": crop,
        "selected_layers": list(getattr(features, "selected_layers", ()) or ()),
    }}
    print(json.dumps(payload))
except Exception as exc:
    payload = {{"ok": False, "error": str(exc), "traceback": traceback.format_exc()}}
    print(json.dumps(payload))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(PROJECT_ROOT),
        )
    except Exception as exc:
        return False, f"Failed to execute probe subprocess: {exc}"

    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "<empty stderr>"
        return False, f"Subprocess returned {proc.returncode}: {stderr}"

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return False, "No output from probe subprocess."

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False, f"Probe output is not JSON: {lines[-1][:300]}"

    if bool(payload.get("ok", False)):
        frames = payload.get("frames_per_clip")
        crop = payload.get("crop_size")
        return True, f"Probe succeeded (frames={frames}, crop={crop})."
    return False, str(payload.get("error", "Unknown probe error"))


def _check_mvp(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    annotation_file = _as_project_path(str(config.get("annotation_file", "")))
    split_dir = _as_project_path(str((config.get("split", {}) or {}).get("dir", "data/splits/mvp/full_60_20_20")))
    videos_root = _as_project_path(str(config.get("videos_root", "")))

    checks.append(_check_result("annotation_file_exists", annotation_file.exists(), str(annotation_file)))
    split_pairs = split_dir / "split_pairs.parquet"
    split_manifest = split_dir / "manifest.json"
    checks.append(_check_result("split_pairs_exists", split_pairs.exists(), str(split_pairs)))
    checks.append(_check_result("split_manifest_exists", split_manifest.exists(), str(split_manifest)))

    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            stats["manifest_stats"] = manifest.get("stats", {})
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
    else:
        checks.append(_check_result("manifest_readable", False, "Manifest file not found."))

    if annotation_file.exists() and split_pairs.exists():
        try:
            import pandas as pd

            frame = pd.read_parquet(split_pairs)
            required_sample_ids: set[str] = set()
            for raw in frame.get("sample_ids_json", []):
                try:
                    required_sample_ids.update(str(item) for item in json.loads(str(raw)))
                except Exception:
                    continue

            rows = load_mvp_rows(annotation_file)
            sample_to_ref: dict[str, str] = {}
            for row in rows:
                sample_to_ref[derive_sample_id(row)] = derive_video_ref(row)

            present = 0
            missing = 0
            missing_refs: list[str] = []
            for sample_id in sorted(required_sample_ids):
                ref = sample_to_ref.get(sample_id, "")
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate or sample_id)

            stats["video_stats"] = {
                "required_samples": len(required_sample_ids),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    else:
        checks.append(_check_result("videos_downloaded", False, "Missing annotation or split_pairs file."))

    return _bundle("dataset", "mvp", None, checks, stats=stats)


def _check_intphys2(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    metadata_file = _as_project_path(str(config.get("metadata_file", "")))
    split_dir = _as_project_path(str((config.get("split", {}) or {}).get("dir", "data/splits/intphys2")))
    videos_root = _as_project_path(str(config.get("videos_root", "")))

    checks.append(_check_result("metadata_file_exists", metadata_file.exists(), str(metadata_file)))
    split_scenes = split_dir / "split_scenes.parquet"
    split_manifest = split_dir / "manifest.json"
    checks.append(_check_result("split_scenes_exists", split_scenes.exists(), str(split_scenes)))
    checks.append(_check_result("split_manifest_exists", split_manifest.exists(), str(split_manifest)))

    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            stats["manifest_stats"] = manifest.get("stats", {})
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
    else:
        checks.append(_check_result("manifest_readable", False, "Manifest file not found."))

    if metadata_file.exists():
        try:
            rows = [normalize_intphys2_row(row) for row in load_intphys2_rows(metadata_file)]
            present = 0
            missing = 0
            missing_refs: list[str] = []
            for row in rows:
                ref = str(row.get("video_path", "")).strip()
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate)
            stats["video_stats"] = {
                "required_samples": len(rows),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    else:
        checks.append(_check_result("videos_downloaded", False, "Metadata file not found."))

    return _bundle("dataset", "intphys2", None, checks, stats=stats)


def _check_ssv2(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    train_file = _as_project_path(str(config.get("train_annotation_file", "")))
    val_file = _as_project_path(str(config.get("val_annotation_file", "")))
    labels_file = _as_project_path(str(config.get("labels_file", "")))
    split_dir = _as_project_path(str((config.get("split", {}) or {}).get("dir", "data/splits/ssv2")))
    videos_root = _as_project_path(str(config.get("videos_root", "")))

    checks.append(_check_result("train_annotation_exists", train_file.exists(), str(train_file)))
    checks.append(_check_result("val_annotation_exists", val_file.exists(), str(val_file)))
    checks.append(_check_result("labels_exists", labels_file.exists(), str(labels_file)))

    split_clips = split_dir / "split_clips.parquet"
    split_manifest = split_dir / "manifest.json"
    checks.append(_check_result("split_clips_exists", split_clips.exists(), str(split_clips)))
    checks.append(_check_result("split_manifest_exists", split_manifest.exists(), str(split_manifest)))

    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            stats["manifest_stats"] = manifest.get("stats", {})
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
    else:
        checks.append(_check_result("manifest_readable", False, "Manifest file not found."))

    if split_clips.exists():
        try:
            import pandas as pd

            frame = pd.read_parquet(split_clips)
            refs: list[str] = []
            if "video_ref" in frame.columns:
                refs = [str(item) for item in frame["video_ref"].tolist()]
            elif "sample_id" in frame.columns:
                refs = [f"{item}.webm" for item in frame["sample_id"].tolist()]

            present = 0
            missing = 0
            missing_refs: list[str] = []
            for ref in refs:
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate)
            stats["video_stats"] = {
                "required_samples": len(refs),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    elif train_file.exists() and val_file.exists():
        try:
            train_rows = load_ssv2_annotations(train_file)
            val_rows = load_ssv2_annotations(val_file)
            refs = [f"{row['id']}.webm" for row in train_rows + val_rows if "id" in row]
            present = 0
            missing = 0
            missing_refs: list[str] = []
            for ref in refs:
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate)
            stats["video_stats"] = {
                "required_samples": len(refs),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    else:
        checks.append(_check_result("videos_downloaded", False, "Missing split_clips and annotations."))

    return _bundle("dataset", "ssv2", None, checks, stats=stats)


def _video_exists(video_ref: str, videos_root: Path) -> tuple[bool, str]:
    ref = str(video_ref or "").strip()
    if not ref:
        return False, "<empty video ref>"

    direct = Path(ref)
    normalized = ref.lstrip("/\\")
    rooted_relative = videos_root / normalized
    local = videos_root / ref

    if rooted_relative.exists():
        return True, str(rooted_relative)
    if direct.is_absolute() and direct.exists():
        return True, str(direct)
    if local.exists():
        return True, str(local)
    return False, str(rooted_relative)


def _check_result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "detail": detail,
    }


def _bundle(
    kind: str,
    name: str,
    variant: str | None,
    checks: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed = any(check.get("status") == "fail" for check in checks)
    payload: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "status": "fail" if failed else "pass",
        "checks": checks,
    }
    if variant is not None:
        payload["variant"] = variant
    if stats:
        payload["stats"] = stats
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(payload)!r}")
    return payload


def _as_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _format_human_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ok = bool(report.get("ok", False))
    summary = report.get("summary", {})
    lines.append("Probe4Physics Health Check")
    lines.append("")
    lines.append(f"Overall: {'PASS' if ok else 'FAIL'}")
    lines.append(
        "Checks: "
        f"total={summary.get('total_checks', 0)} "
        f"passed={summary.get('passed', 0)} "
        f"failed={summary.get('failed', 0)}"
    )
    lines.append("")
    lines.append("Backbones")
    for item in report.get("tested_backbones", []):
        status = str(item.get("status", "unknown")).upper()
        name = str(item.get("name", ""))
        variant = str(item.get("variant", ""))
        lines.append(f"- {name} ({variant}): {status}")

    lines.append("")
    lines.append("Datasets")
    for item in report.get("tested_datasets", []):
        status = str(item.get("status", "unknown")).upper()
        name = str(item.get("name", ""))
        lines.append(f"- {name}: {status}")

    failing = _collect_failing_checks(report)
    lines.append("")
    lines.append("Failing checks")
    if not failing:
        lines.append("- none")
    else:
        for row in failing:
            lines.append(
                f"- {row['scope']} [{row['check']}]: {row['detail']}"
            )

    return "\n".join(lines)


def _collect_failing_checks(report: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for section in ("backbones", "datasets"):
        for item in report.get(section, []):
            name = str(item.get("name", ""))
            variant = str(item.get("variant", "")).strip()
            scope = f"{name} ({variant})" if variant else name
            for check in item.get("checks", []):
                if str(check.get("status")) != "fail":
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "check": str(check.get("name", "")),
                        "detail": str(check.get("detail", "")),
                    }
                )
    return rows
