# Datasets

Tracked configs assume datasets live under `data/` and split artifacts are
materialized into `data/splits/`.

## MVP

Default tracked paths:

- annotation file: `data/annotations/mvp_full.jsonl`
- videos root: `data/videos/mvp`
- split outputs: `data/splits/mvp/full_60_20_20`

Typical flow:

```bash
python run.py init.mvp
python run.py extract.mvp backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

If you store videos elsewhere, override `videos_root` on the CLI or via an
`ops/hpc/` wrapper.

## IntPhys2

Default tracked paths:

- metadata file: `data/annotations/intphys2_metadata.csv`
- videos root: `data/videos/intphys2`
- split outputs: `data/splits/intphys2`

Download flow:

```bash
python run.py download.intphys2
python run.py init.intphys2
python run.py extract.intphys2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

Notes:

- download source: Hugging Face `facebook/IntPhys2`
- optional dependency: `huggingface_hub`
- default download behavior materializes files directly under `videos_root`

## SSv2

Default tracked paths:

- train annotations: `data/annotations/ssv2/train.json`
- validation annotations: `data/annotations/ssv2/validation.json`
- labels: `data/annotations/ssv2/labels.json`
- videos root: `data/videos/ssv2`
- split outputs: `data/splits/ssv2`

Typical flow:

```bash
python run.py init.ssv2
python run.py extract.ssv2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

SSv2 annotations come from the official dataset distribution; they are not
downloaded by this repository.

## Checkpoints

Tracked configs never pin a personal checkpoint path. Pass checkpoint overrides
explicitly, for example:

```bash
python run.py extract.mvp backbone.name=jepa_v1 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py extract.ssv2 backbone.name=ltx_video backbone.kwargs.device=cuda
```

Backbone-specific defaults and variants are defined in `configs/backbones.yaml`.
