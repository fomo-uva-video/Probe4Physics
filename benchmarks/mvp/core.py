from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

from benchmarks.mvp.selection import derive_binary_semantics, resolve_answer_idx


class OfficialIntegrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MVPSample:
    sample_id: str
    pair_id: str
    question: str
    choices: tuple[str, str]
    answer_idx: int
    video_a_ref: str
    video_b_ref: str
    split: str
    plausibility_label: int | None = None
    yes_choice_idx: int | None = None
    no_choice_idx: int | None = None


@dataclass(frozen=True)
class MVPPrediction:
    sample_id: str
    pred_idx: int
    choice_scores: tuple[float, float] | None = None


@dataclass(frozen=True)
class MVPMetrics:
    accuracy: float
    pair_consistency: float
    n_samples: int
    n_pairs: int


class MVPAdapter:
    """
    Loads and executes selected official functions from
    third_party/minimal_video_pairs/tasks/mvp/utils.py:
    - extract_pred
    - mvp_process_results
    - compute_metrics
    """

    REQUIRED_SYMBOLS = (
        "idx_map",
        "get_candidates",
        "extract_pred",
        "mvp_process_results",
        "compute_metrics",
    )

    def __init__(self, official_repo_root: str | Path) -> None:
        self.official_repo_root = Path(official_repo_root)
        self.utils_path = self.official_repo_root / "tasks" / "mvp" / "utils.py"
        if not self.utils_path.exists():
            raise OfficialIntegrationError(
                "Official MVP utils not found. Expected: "
                f"{self.utils_path}. Run: git submodule update --init --recursive"
            )

        self.namespace = self._load_official_namespace()

    def process_result(self, doc: dict[str, Any], prediction_output: str) -> dict[str, Any]:
        fn = self.namespace["mvp_process_results"]
        return fn(doc, [prediction_output])

    def compute_metrics(self, results: list[dict[str, Any]]) -> tuple[float, float]:
        fn = self.namespace["compute_metrics"]
        single_accuracy, pair_accuracy = fn(results)
        return float(single_accuracy), float(pair_accuracy)

    def _load_official_namespace(self) -> dict[str, Any]:
        source = self.utils_path.read_text(encoding="utf-8")
        parsed = ast.parse(source)

        selected_nodes: list[ast.stmt] = []
        for node in parsed.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "idx_map" for target in node.targets):
                    selected_nodes.append(node)
            if isinstance(node, ast.FunctionDef) and node.name in self.REQUIRED_SYMBOLS:
                selected_nodes.append(node)

        module = ast.Module(body=selected_nodes, type_ignores=[])
        compiled = compile(module, filename=str(self.utils_path), mode="exec")
        namespace: dict[str, Any] = {
            "ast": ast,
            "re": re,
            "Union": Union,
        }

        try:
            exec(compiled, namespace)  # noqa: S102 - controlled local source from pinned submodule
        except Exception as exc:
            raise OfficialIntegrationError(
                "Failed to load official MVP utility functions from "
                f"{self.utils_path}: {exc}"
            ) from exc

        missing = [name for name in self.REQUIRED_SYMBOLS if name not in namespace]
        if missing:
            raise OfficialIntegrationError(
                "Official MVP integration missing symbols "
                f"{missing} in {self.utils_path}."
            )

        return namespace


class MVPBenchmark:
    """Single-purpose MVP benchmark adapter with strict official scoring."""

    def __init__(self, official_repo_root: str | Path) -> None:
        self.official_repo_root = Path(official_repo_root)
        self.official = MVPAdapter(self.official_repo_root)

    def load_samples(self, rows: Iterable[dict[str, Any]], split: str) -> list[MVPSample]:
        samples: list[MVPSample] = []

        for row in rows:
            sample_id = self._sample_id(row)
            pair_id = self._pair_id(row, sample_id)
            question = str(row.get("question", "")).strip()
            choices = self._choices(row)
            answer_idx = self._answer_idx(row, choices)
            semantics = None
            try:
                semantics = derive_binary_semantics(choices, answer_idx=answer_idx)
            except ValueError:
                semantics = None
            video_ref = self._video_ref(row)
            samples.append(
                MVPSample(
                    sample_id=sample_id,
                    pair_id=pair_id,
                    question=question,
                    choices=choices,
                    answer_idx=answer_idx,
                    plausibility_label=None if semantics is None else semantics.plausibility_label,
                    yes_choice_idx=None if semantics is None else semantics.yes_choice_idx,
                    no_choice_idx=None if semantics is None else semantics.no_choice_idx,
                    video_a_ref=video_ref,
                    video_b_ref="",
                    split=split,
                )
            )

        samples.sort(key=lambda sample: (sample.pair_id, sample.sample_id))
        return samples

    def evaluate(
        self,
        samples: list[MVPSample],
        predictions: list[MVPPrediction],
    ) -> MVPMetrics:
        pred_by_sample = {pred.sample_id: pred for pred in predictions}
        pair_rows: list[dict[str, Any]] = []

        for sample in samples:
            pred = pred_by_sample.get(sample.sample_id)
            pred_text = self._pred_idx_to_output(pred.pred_idx) if pred is not None else ""

            doc = {
                "video_id": sample.sample_id,
                "question": sample.question,
                "candidates": list(sample.choices),
                "answer": sample.choices[sample.answer_idx],
            }
            processed = self.official.process_result(doc, pred_text)
            pair_rows.append(processed["pair_accuracy"])

        accuracy, pair_consistency = self.official.compute_metrics(pair_rows)

        return MVPMetrics(
            accuracy=accuracy,
            pair_consistency=pair_consistency,
            n_samples=len(samples),
            n_pairs=len({sample.pair_id for sample in samples}),
        )

    @staticmethod
    def _pred_idx_to_output(pred_idx: int) -> str:
        if pred_idx == 0:
            return "Answer: A"
        if pred_idx == 1:
            return "Answer: B"
        return ""

    @staticmethod
    def _sample_id(row: dict[str, Any]) -> str:
        for key in ("sample_id", "id", "video_id"):
            if key in row and str(row[key]).strip():
                return str(row[key]).strip()

        stable_payload = json.dumps(row, sort_keys=True, default=str)
        digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()
        return f"sample_{digest[:16]}"

    @staticmethod
    def _pair_id(row: dict[str, Any], sample_id: str) -> str:
        if "pair_id" in row and str(row["pair_id"]).strip():
            return str(row["pair_id"]).strip()

        if "video_id" in row:
            video_id = str(row["video_id"]).strip()
            chunks = video_id.split("_")
            if len(chunks) > 1:
                return "_".join(chunks[:-1])

        chunks = sample_id.split("_")
        if len(chunks) > 1:
            return "_".join(chunks[:-1])
        return sample_id

    @staticmethod
    def _choices(row: dict[str, Any]) -> tuple[str, str]:
        raw = row.get("candidates", row.get("choices"))
        if raw is None:
            raise ValueError("MVP row is missing 'candidates'/'choices'.")

        if isinstance(raw, str):
            parsed = ast.literal_eval(raw)
        else:
            parsed = raw

        if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
            raise ValueError(f"Expected exactly 2 choices, got: {parsed!r}")

        return str(parsed[0]), str(parsed[1])

    @staticmethod
    def _answer_idx(row: dict[str, Any], choices: tuple[str, str]) -> int:
        if "answer_idx" in row:
            answer_idx = int(row["answer_idx"])
            if answer_idx not in (0, 1):
                raise ValueError(f"Invalid answer_idx={answer_idx}; expected 0/1")
            return answer_idx

        answer = row.get("answer")
        if answer is None:
            raise ValueError("MVP row is missing 'answer' or 'answer_idx'.")

        return resolve_answer_idx(choices, answer)

    @staticmethod
    def _video_ref(row: dict[str, Any]) -> str:
        for key in ("video_path", "video", "video_ref"):
            if key in row and str(row[key]).strip():
                return str(row[key]).strip()
        return ""


def asdict_metrics(metrics: MVPMetrics) -> dict[str, Any]:
    return dataclasses.asdict(metrics)
