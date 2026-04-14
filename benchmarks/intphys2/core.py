from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntPhys2Sample:
    sample_id: str
    scene_id: str
    condition: str
    plausibility: int  # 0 = impossible, 1 = possible
    video_ref: str
    split: str
    difficulty: str = ""
    camera: str = ""


@dataclass(frozen=True)
class IntPhys2Prediction:
    sample_id: str
    pred_idx: int  # 0 = impossible, 1 = possible
    score: float | None = None  # P(possible) used for VOE ranking; higher = more plausible


@dataclass(frozen=True)
class IntPhys2Metrics:
    accuracy: float
    voe_accuracy: float
    n_samples: int
    n_scenes: int
    accuracy_by_condition: dict[str, float]
    voe_by_condition: dict[str, float]


class IntPhys2Benchmark:
    """Evaluates binary plausibility predictions against IntPhys2 ground truth."""

    def load_samples(
        self,
        rows: list[dict[str, Any]],
        split: str,
    ) -> list[IntPhys2Sample]:
        from benchmarks.intphys2.data import derive_sample_id, normalize_intphys2_row

        samples: list[IntPhys2Sample] = []
        for row in rows:
            norm = normalize_intphys2_row(row)
            sample_id = derive_sample_id(norm)
            samples.append(
                IntPhys2Sample(
                    sample_id=sample_id,
                    scene_id=str(norm.get("scene_id", "")),
                    condition=str(norm.get("condition", "")),
                    plausibility=int(norm.get("plausibility", -1)),
                    video_ref=str(norm.get("video_path", "")),
                    split=split,
                    difficulty=str(norm.get("difficulty", "")),
                    camera=str(norm.get("camera", "")),
                )
            )

        samples.sort(key=lambda s: (s.scene_id, s.plausibility, s.sample_id))
        return samples

    def evaluate(
        self,
        samples: list[IntPhys2Sample],
        predictions: list[IntPhys2Prediction],
    ) -> IntPhys2Metrics:
        pred_by_id = {pred.sample_id: pred for pred in predictions}

        correct_total = 0
        correct_by_condition: dict[str, int] = {}
        total_by_condition: dict[str, int] = {}

        for sample in samples:
            pred = pred_by_id.get(sample.sample_id)
            if pred is None:
                continue
            is_correct = int(pred.pred_idx == sample.plausibility)
            correct_total += is_correct

            cond = sample.condition or "unknown"
            correct_by_condition[cond] = correct_by_condition.get(cond, 0) + is_correct
            total_by_condition[cond] = total_by_condition.get(cond, 0) + 1

        n_samples = len(samples)
        accuracy = (correct_total / n_samples * 100.0) if n_samples > 0 else 0.0

        accuracy_by_condition: dict[str, float] = {
            cond: correct_by_condition.get(cond, 0) / total_by_condition[cond] * 100.0
            for cond in sorted(total_by_condition)
            if total_by_condition.get(cond, 0) > 0
        }

        voe_accuracy, voe_by_condition = self._compute_voe(samples, pred_by_id)
        n_scenes = len({sample.scene_id for sample in samples})

        return IntPhys2Metrics(
            accuracy=accuracy,
            voe_accuracy=voe_accuracy,
            n_samples=n_samples,
            n_scenes=n_scenes,
            accuracy_by_condition=accuracy_by_condition,
            voe_by_condition=voe_by_condition,
        )

    def _compute_voe(
        self,
        samples: list[IntPhys2Sample],
        pred_by_id: dict[str, IntPhys2Prediction],
    ) -> tuple[float, dict[str, float]]:
        """
        VOE accuracy: fraction of scenes where every possible video is scored
        strictly higher than every impossible video.

        If a prediction includes a ``score`` (P(possible)), it is used for
        ranking.  Otherwise the binary ``pred_idx`` (0/1) is used.
        """
        scenes_by_id: dict[str, list[IntPhys2Sample]] = defaultdict(list)
        for sample in samples:
            scenes_by_id[sample.scene_id].append(sample)

        voe_correct_total = 0
        n_scenes_total = 0
        voe_correct_by_condition: dict[str, int] = {}
        n_scenes_by_condition: dict[str, int] = {}

        for scene_samples in scenes_by_id.values():
            possible = [s for s in scene_samples if s.plausibility == 1]
            impossible = [s for s in scene_samples if s.plausibility == 0]

            if not possible or not impossible:
                continue

            conditions = {s.condition for s in scene_samples}
            cond = next(iter(conditions)) if len(conditions) == 1 else "mixed"

            def _score(sample: IntPhys2Sample) -> float:
                pred = pred_by_id.get(sample.sample_id)
                if pred is None:
                    return 0.5
                if pred.score is not None:
                    return float(pred.score)
                return float(pred.pred_idx)

            min_possible = min(_score(s) for s in possible)
            max_impossible = max(_score(s) for s in impossible)
            is_correct = int(min_possible > max_impossible)

            voe_correct_total += is_correct
            n_scenes_total += 1
            voe_correct_by_condition[cond] = voe_correct_by_condition.get(cond, 0) + is_correct
            n_scenes_by_condition[cond] = n_scenes_by_condition.get(cond, 0) + 1

        voe_accuracy = (
            voe_correct_total / n_scenes_total * 100.0 if n_scenes_total > 0 else 0.0
        )
        voe_by_condition: dict[str, float] = {
            cond: voe_correct_by_condition.get(cond, 0) / n_scenes_by_condition[cond] * 100.0
            for cond in sorted(n_scenes_by_condition)
            if n_scenes_by_condition.get(cond, 0) > 0
        }

        return voe_accuracy, voe_by_condition


def asdict_metrics(metrics: IntPhys2Metrics) -> dict[str, Any]:
    return dataclasses.asdict(metrics)
