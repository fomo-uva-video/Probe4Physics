# Probe4Physics MVP (Simple Setup)

Configurazione semplificata: ora c'e solo **un file config**:
- `configs/mvp.yaml`

## Comandi

```bash
python run.py init.mvp
python run.py eval.mvp
```

## Cosa fa `init.mvp`
- scarica automaticamente le annotation full (se mancano)
- applica selection (`intuitive_physics` + plausibility yes/no)
- crea split deterministico 60/20/20 per `pair_id`
- salva artifact split in `split.dir`

Artifact generati in `split.dir`:
- `split_pairs.parquet`
- `manifest.json`
- `selection_kept.csv`
- `selection_dropped.csv`
- `selection_report.json`

## Cosa fa `eval.mvp`
- carica solo lo split precomputato da `split.dir`
- fa hash-check annotation vs manifest (hard fail se mismatch)
- usa scoring ufficiale MVP

## Config principale
In `configs/mvp.yaml`:
- `split.dir`: path unico dello split (es. `data/splits/mvp/full_60_20_20`)
- `split.ratios`: `train/val/test`
- `split.group_key`: `pair_id`
- `split.stratify_keys`: `[source, question_template]`

## Environment

```bash
conda env create -f environment.yml
conda activate probe4physics
# if the env already exists, update it:
conda env update -n probe4physics -f environment.yml --prune
```

## Official V-JEPA v1 Adapter Setup

Sync submodules (including official V-JEPA v1 in `third_party/jepa_v1`):

```bash
git submodule update --init --recursive
git -C third_party/jepa_v1 checkout 51c59d518fc63c08464af6de585f78ac0c7ed4d5
```

Download an official V-JEPA v1 checkpoint from:
- https://github.com/facebookresearch/jepa

Manual smoke command (non-CI) for frozen feature extraction:

```bash
python - <<'PY'
import torch
from models import create_adapter

adapter = create_adapter(
    "jepa_v1",
    repo_root="third_party/jepa_v1",
    checkpoint_path="/absolute/path/to/vitl16.pth.tar",
)
clips = torch.randn(1, 3, 16, 224, 224)
features = adapter.extract(clips)
print("layers:", features.selected_layers)
print({k: tuple(v.shape) for k, v in features.tokens_by_layer.items()})
PY
```
