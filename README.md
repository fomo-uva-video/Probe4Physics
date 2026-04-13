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
```
