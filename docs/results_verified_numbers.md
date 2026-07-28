# Verified best-case numbers (from results/*.xlsx, 2026-07-18)

Selection criterion everywhere: **best layer by TEST primary metric** ("best-case"),
matching the convention of the current Table 1. Values are Test Primary Metric
(VOE accuracy for IntPhys2, pair consistency for MVP). Layers are absolute block
indices; LTX layers are `block (noise)`.

## 1. Largest backbones (`main` + LTX-13B) — Table 1

| Model | IP2 Lin | IP2 MLP | IP2 Attn | MVP Lin | MVP MLP | MVP Attn |
|---|---|---|---|---|---|---|
| V-JEPA (ViT-H) | 24 / 50.98 | 32 / 45.10 | 16 / 66.67 | 32 / 48.74 | 32 / 87.26 | 24 / 94.03 |
| V-JEPA 2 (ViT-G) | 40 / 41.18 | 30 / 56.86 | 40 / 78.43 | 40 / **48.53** | 40 / 86.75 | 40 / 93.33 |
| V-JEPA 2.1 (ViT-Gig.) | 48 / 35.29 | 36 / 39.22 | 48 / 60.78 | **36 / 42.26** | **36** / 70.78 | 48 / 93.73 |
| VideoMAE (ViT-H) | 16 / 35.29 | 24 / 35.29 | 24 / 76.47 | 24 / 37.51 | 24 / 64.41 | 24 / 92.01 |
| VideoMAE-v2 (ViT-G) | 40 / 47.06 | 20 / 47.06 | 20 / 62.75 | 20 / **37.71** | 20 / 66.13 | 30 / 91.10 |
| LTX-Video 13B | 24(0.1) / 49.02 | 24(0.2) / 47.06 | 36(0.1)? / 43.14 | 24(0.6) / 49.95 | 24(0.7) / 69.16 | 24(0.5) / 84.33 |

**Bold = differs from the current PDF Table 1** (PDF had: V-JEPA 2 MVP Lin 47.32;
V-JEPA 2.1 MVP Lin 43.88 @ layer 38 [38 is not a probed depth]; V-JEPA 2.1 MVP
MLP @ layer 38; VideoMAE-v2 MVP Lin 38.62).

## 2. Shared ViT-L (`same_L`) + LTX references — Analysis 4 / Figure 3

| Model | IP2 Lin | IP2 MLP | IP2 Attn | MVP Lin | MVP MLP | MVP Attn |
|---|---|---|---|---|---|---|
| V-JEPA | 24 / 47.06 | 24 / 49.02 | 18 / 74.51 | 18 / 47.02 | 18 / 83.72 | 18 / 93.83 |
| V-JEPA 2 | 18 / 49.02 | 18 / 52.94 | 18 / 70.59 | 24 / 54.10 | 24 / 86.96 | 18 / 93.33 |
| V-JEPA 2.1 | 18 / 43.14 | 18 / 50.98 | 24 / 58.82 | 18 / 44.08 | 18 / 72.09 | 24 / 93.63 |
| VideoMAE | 18 / 43.14 | 18 / 41.18 | 24 / 68.63 | 24 / 40.85 | 24 / 63.30 | 24 / 89.89 |
| VideoMAE-v2 | 12 / 41.18 | 12 / 47.06 | 18 / 54.90 | 18 / 40.14 | 18 / 68.86 | 18 / 87.36 |
| LTX-2B | 14(0.3) / 27.45 | 21(0.1) / 27.45 | idx36? / 35.29 | 7(0.1) / 28.31 | 7(0.6) / 40.34 | **MISSING** |
| LTX-13B | 24(0.1) / 49.02 | 24(0.2) / 47.06 | 36(0.1)? / 43.14 | 24(0.6) / 49.95 | 24(0.7) / 69.16 | 24(0.5) / 84.33 |

## 3. Backbone sweep — Analysis 3

Best-case per backbone (Lin / MLP / Attn):

**V-JEPA 2.1**
| Backbone | IntPhys2 | MVP |
|---|---|---|
| ViT-B | 39.22 / 50.98 / 58.82 | 40.75 / 69.87 / 91.41 |
| ViT-L | 43.14 / 50.98 / 58.82 | 44.08 / 72.09 / 93.63 |
| ViT-G | 49.02 / 43.14 / 60.78 | 47.12 / 69.77 / 93.12 |
| ViT-Gigantic | 35.29 / 39.22 / 60.78 | 42.26 / 70.78 / 93.73 |

**VideoMAE-v2**
| Backbone | IntPhys2 | MVP |
|---|---|---|
| ViT-B | 43.14 / 35.29 / 52.94 | 40.34 / 67.85 / 85.74 |
| ViT-L | 41.18 / 47.06 / 54.90 | 40.14 / 68.86 / 87.36 |
| ViT-G | 47.06 / 47.06 / 62.75 | 37.71 / 66.13 / 91.10 |

**LTX-Video** (2B → 13B): IntPhys2 27.45/27.45/35.29 → 49.02/47.06/43.14;
MVP 28.31/40.34/— → 49.95/69.16/84.33.

## 4. Re-run IntPhys2 attentive controls (at new best layers)

| Model | Main | shuffle → Δ% | single → Δ% |
|---|---|---|---|
| V-JEPA (L16) | 66.67 | 39.22 → −41.18 | 15.69 → −76.47 |
| V-JEPA 2 (L40) | 78.43 | 54.90 → −30.00 | 17.65 → −77.50 |
| V-JEPA 2.1 (L48) | 60.78 | 60.78 → 0.00 | 15.69 → −74.19 |
| VideoMAE (L24) | 76.47 | 52.94 → −30.77 | 17.65 → −76.92 |
| VideoMAE-v2 (L20) | 62.75 | 47.06 → −25.00 | 19.61 → −68.75 |
| LTX-13B (36/0.1) | 43.14 | 3.92 → −90.91 | 11.76 → −72.73 |

## Open flags

1. **LTX attentive layer labels are run indices, not layers.** In the primary
   DBs the LTX attentive rows are labeled 34/36/38/39. The controls CSV maps
   (for 13B): 34→noise_0.2_block_24, 36→noise_0.2_block_48,
   38→noise_0.1_block_24, 39→noise_0.1_block_36. Both idx 36 and idx 39 score
   43.14 on IntPhys2, so the paper label "36 (0.1)" is ambiguous — confirm from
   run logs which config it is (control deltas differ: −90.91/−72.73 vs
   −59.09/−81.82). The 2B index mapping (34/36/38/39 on IntPhys2; 27/31/33/39
   on MVP) is unknown to me.
2. **LTX-2B attentive on MVP has no results** (empty rows). Run it or state
   the omission in Analysis 3.
3. **Selection criterion**: Table 1 uses best-by-TEST ("best-case"). Note that
   selecting by validation instead changes several cells (e.g., LTX-13B MVP
   MLP 69.16→66.84, attn 84.33→83.01; V-JEPA 2 IP2 MLP 56.86→47.06). Worth one
   sentence in the methodology defending the best-case (oracle) framing, since
   full per-layer profiles are reported anyway.
4. The old footnote about VideoMAE-v2's anomalous +62.50% single-frame jump is
   obsolete (underfit fixed; main VOE now 62.75) — delete it.
