# Backbone Adapters Design Notes

This document defines the project-wide adapter design so future backbones
(`jepa_v2`, `jepa_v2_1`, `videomae`, `videomae_v2`, etc.) are implemented
consistently, with minimal duplication.

## Goals

- One stable interface for all video backbones.
- Comparable layerwise probing across model families.
- Thin model-specific wrappers; shared logic centralized.
- No implicit behavior differences across adapters.

## Canonical Interface

All adapters must implement `VideoBackboneAdapter.extract(...)` and return
`BackboneFeatures` from `models/registry.py`.

### Input contract

- `clips`: `torch.Tensor` with shape `[B, C, T, H, W]`
- `layer_ids`: optional subset of already configured layers

### Output contract (`BackboneFeatures`)

- `tokens_by_layer: dict[int, torch.Tensor]`
  - Each value shape: `[B, N, D]`
- `pooled_by_layer: dict[int, torch.Tensor]`
  - Each value shape: `[B, D]`
  - Pooling rule: mean over token dimension (`dim=1`)
- `selected_layers: tuple[int, ...]`
  - User-facing 1-based layer ids
- `metadata: dict[str, Any]`
  - Must include enough provenance to reproduce extraction

## Layer Selection Policy

Current protocol from proposal:

- Probe depths: `25%, 50%, 75%, 100%`
- 24-block model -> `6, 12, 18, 24`
- 12-block model -> `3, 6, 9, 12`

Important:

- `100%` means final layer output, not concatenation of all layers.
- Combined-layer evaluations are done downstream in probing logic, not by
  changing adapter output schema.

## Configuration Policy

- Global defaults live in `configs/backbones.yaml`.
- Adapter code should avoid hardcoded experiment defaults.
- Adapter constructor can allow overrides, but zero-config startup should work
  from project config.

## Registry Policy

- Register each adapter via `register_adapter("<name>", factory)`.
- Use `create_adapter("<name>", **kwargs)` from callers.
- Keep adapter naming explicit and stable (`jepa_v1`, `jepa_v2`, ...).

## Runtime Isolation Policy

- If a backbone family can cause import namespace collisions (like JEPA repos
  using top-level `src.*`), enforce runtime guard via
  `enforce_single_jepa_namespace(...)`.
- Run different JEPA families in separate Python processes.

## No-Duplication Rules

When adding a new adapter:

- Reuse shared helpers from `models/registry.py`.
- Keep common config/path/validation logic in shared utilities if duplicated.
- Do not copy-paste large blocks of checkpoint cleaning, layer mapping, or
  metadata assembly across adapters without extracting helper functions.
- Prefer small model-specific wrappers around official code paths.

## Metadata Minimum Keys

Each adapter should include at least:

- `model_name`
- `checkpoint_path`
- `config_path` (if config-driven)
- backbone-specific shape settings (`patch_size`, `tubelet_size`,
  `frames_per_clip`, etc.)
- optional `variant` when preset variants are used

## Testing Requirements (for each new adapter)

- Layer mapping test
- Checkpoint loading/keys test
- Shape contract test for `tokens_by_layer` and `pooled_by_layer`
- Registry registration/factory test
- Missing files/config error test with actionable message

## Future Backbones (Implementation Reminder)

For `videomae` and `videomae_v2`:

- Keep same `BackboneFeatures` output schema
- Keep same layer id semantics (1-based user-facing ids)
- Keep same metadata minimum keys
- Keep same subset behavior (`layer_ids`)
- Keep same error style and test coverage pattern
