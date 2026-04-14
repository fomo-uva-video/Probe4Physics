from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SSv2Sample:
    sample_id: str
    label_idx: int
    label_name: str
    template: str
    video_ref: str
    split: str


@dataclass(frozen=True)
class SSv2Prediction:
    sample_id: str
    pred_idx: int
    scores: tuple[float, ...] | None = None  # per-class scores for top-k evaluation


@dataclass(frozen=True)
class SSv2Metrics:
    top1_accuracy: float
    top5_accuracy: float
    n_samples: int
    n_classes: int
    top1_by_template: dict[str, float]


class SSv2Benchmark:
    """Evaluates multi-class action predictions against SSv2 ground truth."""

    def load_samples(
        self,
        rows: list[dict[str, Any]],
        split: str,
    ) -> list[SSv2Sample]:
        """Load samples from pre-normalised rows (e.g. from split_clips.parquet)."""
        samples: list[SSv2Sample] = []
        for row in rows:
            samples.append(
                SSv2Sample(
                    sample_id=str(row["sample_id"]),
                    label_idx=int(row["label_idx"]),
                    label_name=str(row.get("label_name", "")),
                    template=str(row.get("template", "")),
                    video_ref=str(row.get("video_ref", f"{row['sample_id']}.webm")),
                    split=split,
                )
            )
        samples.sort(key=lambda s: s.sample_id)
        return samples

    def evaluate(
        self,
        samples: list[SSv2Sample],
        predictions: list[SSv2Prediction],
    ) -> SSv2Metrics:
        pred_by_id = {pred.sample_id: pred for pred in predictions}

        top1_correct = 0
        top5_correct = 0
        top1_by_template: dict[str, list[int]] = {}

        for sample in predictions:
            # Iterate predictions so missing samples don't inflate denominator.
            pass

        for sample in samples:
            pred = pred_by_id.get(sample.sample_id)
            if pred is None:
                continue

            is_top1 = int(pred.pred_idx == sample.label_idx)
            top1_correct += is_top1

            if pred.scores is not None and len(pred.scores) >= 5:
                top5_indices = sorted(
                    range(len(pred.scores)), key=lambda i: -pred.scores[i]
                )[:5]
                top5_correct += int(sample.label_idx in top5_indices)
            else:
                # No score vector available — top-5 degrades to top-1.
                top5_correct += is_top1

            tmpl = sample.template or "unknown"
            top1_by_template.setdefault(tmpl, []).append(is_top1)

        n = len(samples)
        top1_accuracy = (top1_correct / n * 100.0) if n > 0 else 0.0
        top5_accuracy = (top5_correct / n * 100.0) if n > 0 else 0.0

        top1_by_template_avg: dict[str, float] = {
            tmpl: sum(vals) / len(vals) * 100.0
            for tmpl, vals in sorted(top1_by_template.items())
            if vals
        }

        n_classes = len({s.label_idx for s in samples})

        return SSv2Metrics(
            top1_accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
            n_samples=n,
            n_classes=n_classes,
            top1_by_template=top1_by_template_avg,
        )


def asdict_metrics(metrics: SSv2Metrics) -> dict[str, Any]:
    return dataclasses.asdict(metrics)
