from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from benchmarks.mvp.features import (
    FeatureConfigError,
    load_feature_cache_for_config,
    resolve_expected_feature_cache_paths,
    run_mvp_feature_extraction,
)
from models.base import BackboneFeatures


class _FakeAdapter:
    frames_per_clip = 2
    crop_size = 4

    def __init__(self) -> None:
        self._call_count = 0

    def extract(self, clips: torch.Tensor, layer_ids=None) -> BackboneFeatures:
        self._call_count += 1
        layer = int(layer_ids[0]) if layer_ids else 12
        pooled = torch.full((clips.shape[0], 4), float(self._call_count), dtype=torch.float32)
        tokens = torch.full((clips.shape[0], 2, 4), float(self._call_count), dtype=torch.float32)
        return BackboneFeatures(
            tokens_by_layer={layer: tokens},
            pooled_by_layer={layer: pooled},
            selected_layers=(layer,),
            metadata={"adapter": "fake"},
        )


class MVPFeatureSemanticIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.official_repo = self.repo_root / "third_party" / "minimal_video_pairs"

    def _write_annotation_fixture(self, path: Path) -> list[dict[str, str]]:
        rows = [
            {
                "video_id": "pair_alpha_0",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "subset/pair_alpha_0.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["Yes", "No"],
                "answer": "Yes",
            },
            {
                "video_id": "pair_alpha_1",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "subset/pair_alpha_1.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["Yes", "No"],
                "answer": "No",
            },
            {
                "video_id": "pair_beta_0",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "subset/pair_beta_0.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["No", "Yes"],
                "answer": "Yes",
            },
            {
                "video_id": "pair_beta_1",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "subset/pair_beta_1.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["No", "Yes"],
                "answer": "No",
            },
        ]

        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return rows

    def _create_video_files(self, videos_root: Path, rows: list[dict[str, str]]) -> None:
        for row in rows:
            video_path = videos_root / str(row["video_path"])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fake")

    def _write_split_artifacts(self, split_dir: Path, annotation_file: Path) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "pair_id": "pair_alpha",
                    "split": "train",
                    "stratum": "all",
                    "source": "intphys",
                    "question_template": "is this video physically plausible after the collision",
                    "n_samples": 2,
                    "sample_ids_json": json.dumps(["pair_alpha_0", "pair_alpha_1"]),
                },
                {
                    "pair_id": "pair_beta",
                    "split": "train",
                    "stratum": "all",
                    "source": "intphys",
                    "question_template": "is this video physically plausible after the collision",
                    "n_samples": 2,
                    "sample_ids_json": json.dumps(["pair_beta_0", "pair_beta_1"]),
                },
            ]
        )
        frame.to_parquet(split_dir / "split_pairs.parquet", index=False)

        digest = hashlib.sha256(annotation_file.read_bytes()).hexdigest()
        (split_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "annotation_sha256": digest,
                    "selection_sha256": "fixture_selection_sha",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _extract_config(self, root: Path, annotation_file: Path, videos_root: Path) -> dict:
        return {
            "annotation_file": str(annotation_file),
            "official_repo_root": str(self.official_repo),
            "videos_root": str(videos_root),
            "cache_dir": str(root / "video_cache"),
            "annotations": {"auto_download": False},
            "split": {"dir": str(root / "splits" / "mvp_train_only")},
            "backbone": {"name": "fake_backbone", "kwargs": {}},
            "feature_cache": {
                "dir": str(root / "features"),
                "split_names": ["train"],
                "layer_ids": [12],
                "include_pooled": True,
                "include_tokens": True,
                "max_samples": 0,
                "force_reextract": True,
            },
        }

    def test_extract_writes_semantic_index_columns_and_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = self._write_annotation_fixture(annotation_file)
            videos_root = tmp_path / "videos"
            self._create_video_files(videos_root, rows)
            self._write_split_artifacts(tmp_path / "splits" / "mvp_train_only", annotation_file)

            extract_cfg = self._extract_config(tmp_path, annotation_file, videos_root)

            with mock.patch("benchmarks.mvp.features.create_adapter", return_value=_FakeAdapter()):
                with mock.patch(
                    "benchmarks.mvp.features._decode_video_clip",
                    return_value=torch.zeros((1, 3, 2, 4, 4), dtype=torch.float32),
                ):
                    result = run_mvp_feature_extraction(extract_cfg)

            self.assertFalse(result["skipped"])
            bundle = load_feature_cache_for_config(extract_cfg)
            manifest = bundle["manifest"]
            index = bundle["index"].sort_values("sample_id").reset_index(drop=True)

            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["targets"]["type"], "semantic_plausibility")
            self.assertIn("plausibility_label", index.columns)
            self.assertIn("yes_choice_idx", index.columns)
            self.assertIn("no_choice_idx", index.columns)

            for row in index.to_dict(orient="records"):
                if int(row["plausibility_label"]) == 1:
                    self.assertEqual(int(row["answer_idx"]), int(row["yes_choice_idx"]))
                else:
                    self.assertEqual(int(row["answer_idx"]), int(row["no_choice_idx"]))

    def test_decode_crop_size_changes_cache_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = self._write_annotation_fixture(annotation_file)
            videos_root = tmp_path / "videos"
            self._create_video_files(videos_root, rows)
            self._write_split_artifacts(tmp_path / "splits" / "mvp_train_only", annotation_file)

            cfg_a = self._extract_config(tmp_path, annotation_file, videos_root)
            cfg_a["decode"] = {"num_frames": 2, "sampling": "uniform", "crop_size": 4}
            cfg_b = self._extract_config(tmp_path, annotation_file, videos_root)
            cfg_b["decode"] = {"num_frames": 2, "sampling": "uniform", "crop_size": 8}

            paths_a = resolve_expected_feature_cache_paths(cfg_a)
            paths_b = resolve_expected_feature_cache_paths(cfg_b)

        self.assertNotEqual(paths_a.signature, paths_b.signature)

    def test_max_samples_changes_cache_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = self._write_annotation_fixture(annotation_file)
            videos_root = tmp_path / "videos"
            self._create_video_files(videos_root, rows)
            self._write_split_artifacts(tmp_path / "splits" / "mvp_train_only", annotation_file)

            cfg_a = self._extract_config(tmp_path, annotation_file, videos_root)
            cfg_a["feature_cache"]["max_samples"] = 0
            cfg_b = self._extract_config(tmp_path, annotation_file, videos_root)
            cfg_b["feature_cache"]["max_samples"] = 2

            paths_a = resolve_expected_feature_cache_paths(cfg_a)
            paths_b = resolve_expected_feature_cache_paths(cfg_b)

        self.assertNotEqual(paths_a.signature, paths_b.signature)

    def test_empty_feature_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = self._write_annotation_fixture(annotation_file)
            videos_root = tmp_path / "videos"
            self._create_video_files(videos_root, rows)
            self._write_split_artifacts(tmp_path / "splits" / "mvp_train_only", annotation_file)

            cfg = self._extract_config(tmp_path, annotation_file, videos_root)
            cfg["feature_cache"]["include_pooled"] = False
            cfg["feature_cache"]["include_tokens"] = False

            with self.assertRaises(FeatureConfigError):
                resolve_expected_feature_cache_paths(cfg)

    def test_invalid_max_samples_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = self._write_annotation_fixture(annotation_file)
            videos_root = tmp_path / "videos"
            self._create_video_files(videos_root, rows)
            self._write_split_artifacts(tmp_path / "splits" / "mvp_train_only", annotation_file)

            cfg = self._extract_config(tmp_path, annotation_file, videos_root)
            cfg["feature_cache"]["max_samples"] = "bad"

            with self.assertRaises(FeatureConfigError):
                resolve_expected_feature_cache_paths(cfg)

    def test_extract_respects_max_samples_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = self._write_annotation_fixture(annotation_file)
            videos_root = tmp_path / "videos"
            self._create_video_files(videos_root, rows)
            self._write_split_artifacts(tmp_path / "splits" / "mvp_train_only", annotation_file)

            extract_cfg = self._extract_config(tmp_path, annotation_file, videos_root)
            extract_cfg["feature_cache"]["max_samples"] = 2

            with mock.patch("benchmarks.mvp.features.create_adapter", return_value=_FakeAdapter()):
                with mock.patch(
                    "benchmarks.mvp.features._decode_video_clip",
                    return_value=torch.zeros((1, 3, 2, 4, 4), dtype=torch.float32),
                ):
                    result = run_mvp_feature_extraction(extract_cfg)

            self.assertFalse(result["skipped"])
            bundle = load_feature_cache_for_config(extract_cfg)
            manifest = bundle["manifest"]
            index = bundle["index"].sort_values("feature_index").reset_index(drop=True)

            self.assertEqual(len(index), 2)
            self.assertEqual(manifest["features"]["max_samples"], 2)
            self.assertEqual(index["sample_id"].tolist(), ["pair_alpha_0", "pair_alpha_1"])


if __name__ == "__main__":
    unittest.main()
