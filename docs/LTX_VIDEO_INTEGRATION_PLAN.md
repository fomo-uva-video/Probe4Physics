# LTX-Video Integration Plan (Proposal)

## Objective

Integrate Hugging Face LTX-Video as a pullable backbone option in Probe4Physics, following the existing pipeline contract:

1. init splits
2. extract frozen features
3. train linear probe
4. eval linear probe

This plan is intentionally implementation-first for extraction/probing (not full video generation workflows), so it fits the current benchmark architecture.

## Constraints to Preserve

Based on current repository patterns and docs:

- Keep the canonical adapter interface in `models/base.py` and `models/registry.py`.
- Keep output schema as `BackboneFeatures` with:
  - `tokens_by_layer: dict[int, Tensor[B, N, D]]`
  - `pooled_by_layer: dict[int, Tensor[B, D]]`
  - `selected_layers: tuple[int, ...]`
  - reproducibility metadata.
- Keep config-driven defaults in `configs/backbones.yaml`.
- Keep extraction path unchanged (`extract.*` commands should work with only config/override changes).
- Keep test style consistent with `tests/test_videomae_adapter.py`, `tests/test_jepa_v*_adapter.py`.
- Follow Experiment Guide flow and Hydra override behavior (including `+` for newly introduced kwargs).

## Scope

### In Scope

- Add a new backbone key: `ltx_video`.
- Add adapter + registry wiring so `create_adapter("ltx_video", **kwargs)` works.
- Support model pull from Hugging Face during adapter initialization (with optional cache dir override).
- Add smoke script and tests.
- Add recipe entries for linear probing with LTX backbone across MVP, IntPhys2, SSv2.
- Add docs commands for pull + extract + train + eval.

### Out of Scope (for this phase)

- Full text/image/video generation workflows from the LTX repo.
- Adding new benchmark tasks specific to generation quality.
- Architectural refactor of training/eval loops.

## Jobs Setup Impact (After Reviewing jobs/setup)

Current scripts in [jobs/setup](../jobs/setup) are mostly hardwired for JEPA v1 and need a small extension for LTX validation on cluster:

1. [jobs/setup/mvp_linear_test.sh](../jobs/setup/mvp_linear_test.sh)
   - currently pins `name=mvp.jepa_v1.linear`, `backbone.name=jepa_v1`, and JEPA checkpoint args.
   - should be parameterized so the same smoke job can run `ltx_video` without copy-paste script forks.
2. [jobs/setup/intphys2_init_extract.sh](../jobs/setup/intphys2_init_extract.sh)
   - currently runs `extract.intphys2` with `backbone.name=jepa_v1` and JEPA checkpoint arguments.
   - should support `BACKBONE_NAME` and backbone-specific kwargs for LTX extraction smoke.
3. [jobs/setup/setup_env.sh](../jobs/setup/setup_env.sh)
   - already good for env updates from `environment-gpu.yml`; no logic changes needed beyond dependency additions in env file.
4. [jobs/setup/setup_mvp_data.sh](../jobs/setup/setup_mvp_data.sh)
   - mostly data bootstrap + split init; no LTX-specific logic required.
5. [jobs/setup/mvp_linear.sh](../jobs/setup/mvp_linear.sh)
   - contains `<you>` placeholders and is less reusable; either parameterize it or mark as legacy in docs.

Conclusion:

- Unit tests do not require job script changes.
- Cluster validation and reproducible smoke runs do require job script updates.

## Technical Plan

## Phase 0: Feasibility Spike (Mandatory Gate)

Reason: LTX-Video is a diffusion generation model, while this repo expects deterministic feature extraction from input clips.

Tasks:

1. Validate which internal representation is stable and suitable as probe features:
   - candidate A: transformer hidden states from the LTX denoiser at fixed timestep/noise policy.
   - candidate B: VAE/latent encoder outputs as token features.
2. Confirm tensor shape can be mapped to `[B, N, D]` consistently.
3. Confirm inference can run in no-grad mode with frozen weights on target hardware.

Exit criteria:

- One selected feature path with deterministic behavior and reasonable runtime.
- Clear mapping from model depth to user-facing layer ids.

If this gate fails, fallback is to postpone adapter and only add model pull tooling.

## Phase 1: Config and Dependencies

Files to update:

- `configs/backbones.yaml`
- `environment.yml`
- `environment-gpu.yml`

Changes:

1. Add `ltx_video` section in `configs/backbones.yaml`:
   - `default_variant`
   - `default_relative_depths`
   - `model_block_depths`
   - `variants.<name>.hf_model_id` and shape defaults (`frames_per_clip`, `crop_size`, `patch_size` if applicable)
2. Add required runtime libraries (exact versions after spike confirmation), likely:
   - `diffusers`
   - `transformers`
   - `huggingface_hub`
   - `safetensors`
   - `accelerate`

Notes:

- Keep existing environments reproducible; avoid over-constraining versions unless required.
- Prefer parity between CPU and GPU env manifests where feasible.

## Phase 2: Adapter Implementation

New file:

- `models/ltx_video_adapter.py`

Reference patterns:

- `models/videomae_adapter.py` for Hugging Face loading style and config handling.
- `models/ADAPTERS_DESIGN.md` for contract and testing requirements.

Core responsibilities:

1. Load config section for `ltx_video` from `configs/backbones.yaml`.
2. Resolve variant and `hf_model_id`.
3. Pull model via HF API (`from_pretrained(...)`) with optional `hf_cache_dir`.
4. Freeze model parameters and run inference in `torch.no_grad()`.
5. Implement layer mapping helper:
   - relative depths -> 1-based layer ids.
6. Implement `extract(clips, layer_ids=None)`:
   - validate input shape `[B, C, T, H, W]`
   - return `BackboneFeatures` with subset behavior identical to existing adapters.
7. Populate metadata keys minimally:
   - `model_name`, `hf_model_id`, `config_path`, `variant`, shape parameters.

## Phase 3: Registry and Package Wiring

Files to update:

- `models/__init__.py`
- (possibly) `models/registry.py` only if helper reuse is needed.

Changes:

1. Export adapter class + factory from package init.
2. Register adapter under stable key `ltx_video`.

## Phase 4: Smoke and User-Facing Pull Path

New file:

- `experiments/smoke_ltx_video.py`

Behavior:

1. Parse args (`--variant`, `--hf-cache-dir`, `--device`, `--batch-size`).
2. Instantiate `create_adapter("ltx_video", ...)`.
3. Run one synthetic forward pass and print selected layers/shapes/metadata.

This becomes the immediate "pull now" command because initialization triggers HF download.

Additionally, add a SLURM wrapper script for cluster smoke validation:

- `jobs/setup/ltx_smoke.sh` (new)
   - activates `probe4physics-gpu`
   - runs `python experiments/smoke_ltx_video.py ...`
   - runs a short extraction smoke (MVP or IntPhys2) with `backbone.name=ltx_video`
   - captures logs under `jobs/setup/out/`

## Phase 5: Experiments Registry Recipes

File to update:

- `experiments/registry.py`

Add recipes:

- `mvp.ltx_video.linear`
- `intphys2.ltx_video.linear`
- `ssv2.ltx_video.linear`

Each recipe should override:

- `backbone.name=ltx_video`
- optional recommended default variant/device overrides.

No run.py command additions are required if we keep existing extract/train/eval commands and just use recipe/config overrides.

## Phase 6: Tests

New tests:

- `tests/test_ltx_video_adapter.py`

Test matrix (aligned with existing style):

1. layer mapping correctness
2. missing/invalid config behavior
3. unknown variant behavior
4. registry presence + factory creation
5. extract shape contract (`tokens_by_layer`, `pooled_by_layer`)
6. subset layer selection behavior
7. metadata minimum keys
8. import/dependency error messages are actionable

Potential adjustments:

- `tests/test_run_commands.py` only if experiment list assertions are expanded.

Operational test execution (cluster):

- add a lightweight job entrypoint for unit tests, or include test invocation in `ltx_smoke.sh`:
   - `python -m unittest tests.test_ltx_video_adapter tests.test_run_commands`

## Phase 7: Documentation Updates

Files to update:

- `docs/EXPERIMENT_GUIDE.md`
- `README.md`

Add:

1. LTX checkpoint/model readiness section (HF access + cache path).
2. Extract/train/eval command examples using `backbone.name=ltx_video`.
3. Full recipe examples via `exp.run name=<ltx recipe>`.
4. Troubleshooting notes:
   - missing HF auth/access
   - OOM guidance (variant/downscale/batch-size)
   - mismatched frame/resolution constraints.
5. Jobs section update:
   - which scripts are generic vs JEPA-specific
   - how to launch LTX smoke/test jobs on Snellius.

## Phase 8: jobs/setup Script Updates

Files to update:

- [jobs/setup/mvp_linear_test.sh](../jobs/setup/mvp_linear_test.sh)
- [jobs/setup/intphys2_init_extract.sh](../jobs/setup/intphys2_init_extract.sh)
- optionally [jobs/setup/mvp_linear.sh](../jobs/setup/mvp_linear.sh) (or document as legacy)
- new jobs/setup/ltx_smoke.sh

Required changes:

1. Introduce generic backbone env knobs:
   - `BACKBONE_NAME` (default `jepa_v1`)
   - `BACKBONE_VARIANT` (model-dependent)
   - `BACKBONE_DEVICE` (default `cuda`)
   - `BACKBONE_EXTRA_OVERRIDES` for adapter-specific Hydra args.
2. Select experiment recipe dynamically:
   - `EXPERIMENT_NAME` defaulting to current JEPA recipe, override to `mvp.ltx_video.linear`.
3. Conditional checkpoint handling:
   - keep `CKPT` required only for JEPA families.
   - skip JEPA checkpoint assertions when `BACKBONE_NAME=ltx_video`.
4. Add smoke-mode fast defaults for LTX:
   - fewer frames, smaller batch, `feature_cache.include_tokens=false` for quick verification.
5. Add explicit test command in job flow (optional flag):
   - `RUN_UNIT_TESTS=true` triggers targeted unittest invocation.

## Proposed Execution Order

1. Phase 0 feasibility spike
2. Phase 1 config/deps
3. Phase 2 adapter
4. Phase 3 registry wiring
5. Phase 4 smoke pull script
6. Phase 5 recipe registration
7. Phase 6 tests
8. Phase 7 docs
9. Phase 8 jobs/setup updates

## Validation Commands (Post-Implementation)

Environment:

```bash
conda env update -n probe4physics -f environment.yml --prune
```

Smoke pull:

```bash
python experiments/smoke_ltx_video.py --device cpu
```

MVP extraction with LTX:

```bash
python run.py extract.mvp backbone.name=ltx_video +backbone.kwargs.variant=<variant_name>
```

Full MVP recipe:

```bash
python run.py exp.run name=mvp.ltx_video.linear backbone.kwargs.device=cuda
```

Tests:

```bash
python -m unittest tests.test_ltx_video_adapter
python -m unittest tests.test_run_commands
```

Snellius smoke/test job:

bash jobs/setup/ltx_smoke.sh

## Risks and Mitigations

1. Representation mismatch (generation model vs probe feature needs)
   - Mitigation: mandatory Phase 0 spike and adapter-level deterministic extraction policy.
2. Heavy VRAM/runtime cost
   - Mitigation: default to smaller/distilled variant; recommend smaller frame count and batch size for extraction.
3. HF access/license constraints
   - Mitigation: actionable error messages and explicit docs for token/login/license acceptance.
4. Dependency churn in diffusers/LTX APIs
   - Mitigation: pin minimal compatible versions after spike and include smoke test in CI.

## Definition of Done

- `create_adapter("ltx_video")` initializes and pulls model successfully.
- `extract.*` commands run with `backbone.name=ltx_video` without pipeline code changes.
- Feature cache produced in standard format and reused by `exp.run` cache checks.
- Adapter tests pass with coverage of shape, registry, error paths, and metadata.
- Docs include copy-paste commands matching Experiment Guide style.
- Cluster smoke job runs with LTX backbone and emits logs/artifacts successfully.

## Approval Request

If approved, implementation will start with Phase 0 (feasibility spike) and then proceed through Phases 1-8 in order.