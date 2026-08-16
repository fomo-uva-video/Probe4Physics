from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

import run
from benchmarks.intphys2.features import (
    _is_valid_cache as is_valid_intphys2_feature_cache,
    resolve_expected_feature_cache_paths as resolve_intphys2_feature_cache_paths,
)
from benchmarks.mvp.features import (
    _feature_cfg as mvp_feature_cfg,
    _find_compatible_feature_cache as find_compatible_mvp_feature_cache,
    _is_valid_cache as is_valid_mvp_feature_cache,
    resolve_expected_feature_cache_paths as resolve_mvp_feature_cache_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact feature caches used by seed manifests.")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    args = parser.parse_args()

    checks: dict[str, dict[str, Any]] = {}
    seen_expected: set[tuple[str, str]] = set()
    for manifest in args.manifest:
        for row in _read_csv(manifest):
            dataset = row["dataset_hydra"].strip()
            overrides = _manifest_overrides(row)
            cfg = run._compose_config("mvp" if dataset == "mvp" else "intphys2", overrides)
            config = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(config, dict):
                raise TypeError(f"Resolved config for {row['run_id']} is not a dict")
            resolution = "exact"
            if dataset == "intphys2":
                expected_paths = resolve_intphys2_feature_cache_paths(config)
                paths = expected_paths
                expected_key = (dataset, str(expected_paths.cache_dir))
                if expected_key in seen_expected:
                    continue
                seen_expected.add(expected_key)
                valid = is_valid_intphys2_feature_cache(paths)
            elif dataset == "mvp":
                expected_paths = resolve_mvp_feature_cache_paths(config)
                paths = expected_paths
                expected_key = (dataset, str(expected_paths.cache_dir))
                if expected_key in seen_expected:
                    continue
                seen_expected.add(expected_key)
                valid = is_valid_mvp_feature_cache(paths)
                if not valid:
                    compatible = find_compatible_mvp_feature_cache(
                        config,
                        exact_paths=expected_paths,
                        feature_cfg=mvp_feature_cfg(config),
                    )
                    if compatible is not None and is_valid_mvp_feature_cache(compatible):
                        paths = compatible
                        valid = True
                        resolution = f"compatible-for-{expected_paths.signature}"
            else:
                raise ValueError(f"Unsupported dataset_hydra={dataset!r}")
            key = str(paths.cache_dir)
            checks.setdefault(
                key,
                {
                    "dataset": dataset,
                    "backbone_name": row["backbone_name"],
                    "backbone_variant": row["backbone_variant"],
                    "signature": str(paths.signature),
                    "cache_dir": str(paths.cache_dir),
                    "valid": bool(valid),
                    "resolution": locals().get("resolution", "exact"),
                    "example_run_id": row["run_id"],
                },
            )
            if not valid:
                checks[key]["valid"] = False

    rows = sorted(checks.values(), key=lambda item: (item["dataset"], item["backbone_name"], item["backbone_variant"], item["cache_dir"]))
    print(json.dumps({"cache_groups": rows, "ready": sum(1 for r in rows if r["valid"]), "total": len(rows)}, indent=2, sort_keys=True))
    missing = [row for row in rows if not row["valid"]]
    if missing:
        raise SystemExit(f"Missing/invalid feature cache groups: {len(missing)}/{len(rows)}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_overrides(row: dict[str, str]) -> list[str]:
    raw = row.get("hydra_overrides_json", "").strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid hydra_overrides_json for {row.get('run_id')!r}")
    return parsed


if __name__ == "__main__":
    main()
