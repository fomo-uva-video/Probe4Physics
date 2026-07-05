"""Shared plotting colors for Probe4Physics notebooks."""

FAMILY_COLORS = {
    "V-JEPA": "#1B6CA8",
    "VideoMAE": "#C44E52",
    "LTX-Video": "#2E8B57",
    "Diffusion video model": "#2E8B57",
}

MODEL_COLORS = {
    "jepa_v1": "#1F77B4",
    "jepa_v2": "#005A8D",
    "jepa_v2_1": "#08306B",
    "videomae": "#D95F02",
    "videomae_v2": "#A63603",
    "ltx_video": FAMILY_COLORS["LTX-Video"],
}

PROBE_COLORS = {
    "linear": "#009E73",
    "mlp": "#E69F00",
    "temporal_attn": "#CC79A7",
}

PROBE_LABEL_COLORS = {
    "Linear": PROBE_COLORS["linear"],
    "linear": PROBE_COLORS["linear"],
    "MLP": PROBE_COLORS["mlp"],
    "Attentive": PROBE_COLORS["temporal_attn"],
    "attentive": PROBE_COLORS["temporal_attn"],
}

BACKBONE_COLORS = {
    "ViT-B/16": "#A6CEE3",
    "ViT-L/16": "#1F78B4",
    "ViT-G/16": "#084081",
    "ViT-H/16": "#4C78A8",
    "ViT-Gigantic/16": "#081D58",
    "LTX-2B": "#74C476",
    "LTX-13B": "#006D2C",
}

DATASET_COLORS = {
    "mvp": "#4C78A8",
    "intphys2": "#F58518",
}
