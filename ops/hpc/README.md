# HPC Wrappers

`ops/hpc/` contains optional cluster launchers for the same pipeline documented
in the top-level `README.md`.

The primary workflow is still:

1. prepare datasets and checkpoints
2. run `init.*`
3. run `extract.*`
4. run `train.probe.*` or `train_eval.probe.*`
5. run `eval.probe.*`

Cluster wrappers exist to submit those steps to a scheduler, not to define a
different conceptual pipeline.

## Layout

- `ops/hpc/extract/`: extraction wrappers and shared shell helpers
- `ops/hpc/train/`: probe training wrappers
- `ops/hpc/init/`: environment, checkpoint, and health wrappers
- `ops/hpc/baseline/`: scheduler wrappers for control-baseline runs
- `ops/hpc/examples/`: standalone scheduler examples retained for reference

## Usage Pattern

Wrappers should provide cluster-specific resources plus path/device overrides.
Tracked YAMLs remain portable and repo-relative by default.

Typical examples:

```bash
sbatch ops/hpc/extract/mvp/run_extract.sh
sbatch ops/hpc/train/intphys2/linear/run_train.sh
sbatch ops/hpc/init/health/full_health.sh
```

## Path Overrides

When using these wrappers, prefer setting environment variables or explicit CLI
overrides inside the wrapper invocation instead of editing tracked config files.

Examples of wrapper-owned concerns:

- scheduler resources
- GPU and partition selection
- scratch and checkpoint mount points
- user- or cluster-specific cache locations

Tracked source trees under `ops/hpc/` must stay template-only. Runtime `.out`,
`.err`, and nested `output/` dumps belong in ignored artifact locations, not in
tracked source.
