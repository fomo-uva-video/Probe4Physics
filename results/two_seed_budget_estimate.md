# Two-Seed Budget Estimate for Main Probe Robustness

Date: 2026-08-05

## Scope

This estimates the cost of adding **two more seeds** for the main biggest-backbone experiment.

Included:

- Datasets: **IntPhys2** and **MVP**.
- Models: V-JEPA v1, V-JEPA2, V-JEPA2.1, VideoMAE, VideoMAE-v2, LTX-Video.
- Probes: linear, MLP, temporal attention.
- Feature caches are assumed to already exist and are reused.
- Primary estimate assumes **selected-config reruns**: the already selected layer and hyperparameters are rerun for two new seeds.

Excluded:

- Feature extraction.
- Failed, cancelled, or exploratory jobs.
- Repeating the full layer/LR/Optuna selection procedure, except in the separate "full resweep" table below.

SBU formula used from Slurm `AllocTRES`:

- A100 jobs: `wall_hours * 128`.
- H100 jobs: `wall_hours * 192`.
- Rome CPU jobs observed in this project: `wall_hours * 16`.

## Recommended Budget

For two additional selected-config seeds, reserve about:

| Budget item | Expected | Conservative reserve |
| --- | ---: | ---: |
| GPU hours | 34.8 | 38.3 |
| SBU credits | 4,577 | 5,053 |

The GPU cost is almost entirely temporal attention. Linear and MLP should be run on CPU and add only a small SBU overhead.

## Budget by Dataset and Probe

| Dataset | Probe | Models | GPU hours, expected | SBU, expected | SBU, conservative | Basis |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| IntPhys2 | Linear | 6 | 0.00 | 2 | 5 | Inferred from CPU linear wrappers and MLP timing; no retained linear Slurm logs found. |
| IntPhys2 | MLP | 6 | 0.00 | 4 | 8 | Historical CPU MLP Optuna logs, plus inferred LTX overhead. |
| IntPhys2 | Temporal attention | 6 | 7.64 | 1,079 | 1,187 | Exact attentive matrix accounting normalized from 16 tasks to 2 selected-config seeds. |
| MVP | Linear | 6 | 0.00 | 6 | 12 | Inferred from CPU linear wrappers and MVP MLP timing; no retained linear Slurm logs found. |
| MVP | MLP | 6 | 0.00 | 12 | 20 | Historical CPU MLP Optuna logs, plus inferred LTX overhead. |
| MVP | Temporal attention | 6 | 27.14 | 3,474 | 3,821 | Exact attentive matrix accounting normalized from 16 tasks to 2 selected-config seeds. |
| **Total** | **All probes** | **36 runs per seed, 72 seed-runs total** | **34.78** | **4,577** | **5,053** | Training/eval only, no extraction. |

## Evidence Used

### Temporal Attention

The recent attentive control jobs are the best timing evidence because they used the same fixed-LR matrix style as the main attentive experiment: 4 selected layers x 4 learning rates = 16 Slurm array tasks per model. I normalized each completed matrix by `2 / 16` to estimate two selected-config seed reruns.

| Dataset | Model | Slurm job | Matrix GPU hours | Matrix SBU | Two-seed selected GPU hours | Two-seed selected SBU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IntPhys2 | V-JEPA v1 | 24990981 | 8.539 | 1,639.5 | 1.067 | 204.9 |
| IntPhys2 | V-JEPA2 | 25007291 | 13.108 | 1,677.9 | 1.639 | 209.7 |
| IntPhys2 | V-JEPA2.1 | 25007290 | 23.422 | 2,998.0 | 2.928 | 374.7 |
| IntPhys2 | VideoMAE | 24997906 | 5.204 | 666.1 | 0.650 | 83.3 |
| IntPhys2 | VideoMAE-v2 | 24997905 | 6.829 | 874.2 | 0.854 | 109.3 |
| IntPhys2 | LTX-Video | 25021478 | 4.031 | 774.0 | 0.504 | 96.7 |
| MVP | V-JEPA v1 | 24996212 | 41.821 | 5,353.1 | 5.228 | 669.1 |
| MVP | V-JEPA2 | 25014962 | 40.918 | 5,237.5 | 5.115 | 654.7 |
| MVP | V-JEPA2.1 | 25014971 | 42.509 | 5,441.2 | 5.314 | 680.1 |
| MVP | VideoMAE | 25014973 | 16.502 | 2,112.2 | 2.063 | 264.0 |
| MVP | VideoMAE-v2 | 25014972 | 19.125 | 2,448.0 | 2.391 | 306.0 |
| MVP | LTX-Video | 25022262 | 56.232 | 7,197.7 | 7.029 | 899.7 |

Totals from the completed matrices:

| Dataset | Matrix GPU hours | Matrix SBU | Two-seed selected GPU hours | Two-seed selected SBU |
| --- | ---: | ---: | ---: | ---: |
| IntPhys2 | 61.13 | 8,630 | 7.64 | 1,079 |
| MVP | 217.11 | 27,790 | 27.14 | 3,474 |
| **Total** | **278.24** | **36,419** | **34.78** | **4,552** |

### MLP

Historical main MLP logs were found for five non-LTX models per dataset under `jobs/train/*/mlp/output/training`. These are CPU jobs with `ENABLE_OPTUNA=true`, `OPTUNA_N_TRIALS=20`, and four layers per model.

| Dataset | Retained jobs | Full Optuna wall time | Full Optuna SBU | Per-model SBU |
| --- | ---: | ---: | ---: | ---: |
| IntPhys2 | 5 | 2.10 CPU wall-hours | 33.6 | 6.7 |
| MVP | 5 | 12.84 CPU wall-hours | 205.4 | 41.1 |
| **Total** | **10** | **14.93 CPU wall-hours** | **238.9** | **23.9** |

Job IDs: 22594073, 22594074, 22594076, 22594085, 22594091, 22594123, 22594124, 22594132, 22594133, 22594139.

No retained LTX MLP timing logs were found. The estimate above therefore adds a small conservative LTX overhead for selected-config reruns only. The recent `24594802` MLP control was intentionally not used as the main estimate because it completed in seconds and is a lower-bound smoke/control case, not representative of the original main Optuna sweep.

Important engineering note: some MVP LTX MLP layer wrappers request `gpu_a100` even though `PROBE_DEVICE=cpu`. Seed reruns should use CPU/Rome wrappers or patched CPU submissions, otherwise they will waste GPU allocation.

### Linear

No retained completed linear stdout/stderr or Slurm accounting records with recognizable linear job names were found. The wrappers show linear uses pooled cached features, CPU/Rome, `ENABLE_OPTUNA=true`, and `OPTUNA_N_TRIALS=20`, matching the MLP orchestration but with a smaller probe. The selected-config estimate is therefore inferred as cheaper than MLP and rounded up for Slurm/job overhead.

## If We Repeat Full Sweeps Instead

This is not the recommended "more seeds" protocol, but it is the cost if the selection procedure itself is repeated for each new seed.

| Scenario for two new seeds | GPU hours | SBU estimate | Notes |
| --- | ---: | ---: | --- |
| Selected configs only | 34.8 | about 4.6k | Recommended. Measures seed variability of the final chosen configurations. |
| Temporal attention full LR matrices | 556.5 | 72.8k | Exact from completed attentive matrices multiplied by two seeds. |
| MLP full Optuna sweeps | 0 if CPU | about 0.6k to 1.4k | Non-LTX evidence is exact; range depends on whether LTX is treated as a small selected subset or all 40 slots. |
| Linear full Optuna sweeps | 0 if CPU | about 0.2k to 0.7k | Inferred from MLP and linear wrappers. |
| Full reselection for all probes | 556.5 | about 73.5k to 75k | Dominated by temporal attention. |

## Recommendation

Run **selected-config reruns only** for the two extra seeds. That is scientifically the clean robustness check for the reported result: it estimates variance of the chosen model/probe configuration without paying again for hyperparameter and layer selection. Budget **40 GPU-hours and 5.1k SBU** to be safe.
