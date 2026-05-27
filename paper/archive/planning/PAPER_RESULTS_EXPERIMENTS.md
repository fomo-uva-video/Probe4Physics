# Paper Results Experiments

This note turns the current result space into four paper-ready experiments.

The goal is not to lock in a single visualization today. The goal is to define:

- the question each experiment answers
- the exact slice of results to use
- the tables we can generate
- multiple plot variants for each experiment
- the tradeoffs of each variant

This file is meant to be a planning document for generating candidate tables and figures later.

## Current Assumption

For now, use `MLP` as the main comparable probe whenever a comparison must include all model families, because `LTX attentive` is still incomplete.

When all attentive runs are available, Experiment 1 can be switched from `MLP` to `attentive` if we decide that the headline comparison should use the strongest probe instead of the most complete probe.

## Global Reporting Rules

- Use `pair consistency` as the primary metric for `MVP`.
- Use `VOE accuracy` as the primary metric for `IntPhys2`.
- Use clip `accuracy` as a secondary metric for both datasets.
- For main-task comparisons, each `model x probe x dataset` entry should use the best layer selected on validation and then reported on test.
- For probe-comparison experiments, each probe should be allowed to use its own best layer.
- For control experiments, use the same checkpoint and the same best layer selected from the original task.
- For the main paper, use control results with the original labels.
- `all_true` and `all_false` can be generated as secondary analyses, but should not be the main control evaluation.

## Four Main Experiments

1. Which model retains the most intuitive-physics understanding?
2. Where is the physics signal localized in depth?
3. Which probe extracts the physics signal best?
4. What happens under temporal control experiments?

---

## Experiment 1

### Working Title

Which model retains the most intuitive-physics understanding?

### Core Question

Among the pretrained video backbones, which ones contain the most accessible intuitive-physics information under a strong readout?

### Why This Experiment Matters

This is the headline comparison. It tells the reader whether different pretraining paradigms lead to different amounts of accessible physics knowledge.

It also sets up an important narrative point:

- the ranking is not necessarily the same on `MVP` and `IntPhys2`
- therefore the answer depends on what kind of intuitive-physics judgment we ask the model to make

### Data Slice

- Main version for now: `MLP`, best validation layer, report test
- Later possible version: `attentive`, best validation layer, report test
- Datasets: `MVP` and `IntPhys2`
- Models:
  - `V-JEPA`
  - `V-JEPA 2`
  - `V-JEPA 2.1`
  - `VideoMAE`
  - `VideoMAE-v2`
  - `LTX-Video`

### Table Variants

#### Table 1A: Headline Cross-Benchmark Comparison

Rows:

- one row per model

Columns:

- `Model family`
- `Pretraining objective`
- `Best MVP layer`
- `MVP pair consistency`
- `MVP accuracy`
- `Best IntPhys2 layer`
- `IntPhys2 VOE`
- `IntPhys2 accuracy`

Why generate it:

- this is the most standard headline table
- it gives exact numbers
- it works even without a main figure

#### Table 1B: Compact Leaderboard Table

Rows:

- one row per model

Columns:

- `Model`
- `MVP primary metric`
- `IntPhys2 primary metric`
- `MVP rank`
- `IntPhys2 rank`

Why generate it:

- visually lighter than Table 1A
- makes benchmark-dependent rank changes very easy to read

#### Table 1C: Family Summary Table

Rows:

- one row per pretraining family

Columns:

- `Family`
- `Representative model(s)`
- `Best result on MVP`
- `Best result on IntPhys2`
- `Narrative takeaway`

Why generate it:

- more interpretive
- useful if the paper wants a stronger story and fewer raw numbers in the main text

### Plot Variants

#### Plot 1A: Two-Panel Cleveland Dot Plot

Layout:

- `2 columns = 2 datasets`
- left: `MVP`
- right: `IntPhys2`
- one dot per model

Encoding:

- x-axis: primary metric
- y-axis: model
- annotate exact numbers next to dots

Why generate it:

- clean and publication-friendly
- better than bars when differences are subtle
- makes cross-benchmark reordering very visible

#### Plot 1B: Grouped Bar Chart

Layout:

- x-axis: model
- two bars per model: one for each dataset

Why generate it:

- familiar and easy to read
- useful if the visual style of the paper is more conventional

Weakness:

- the two metrics are on different semantic tasks, so grouped bars can feel less conceptually crisp than separate panels

#### Plot 1C: Slopegraph of Rank Change

Layout:

- left side: rank on `MVP`
- right side: rank on `IntPhys2`
- one line per model

Why generate it:

- great if the key story is that benchmark choice changes the ranking
- visually memorable

Weakness:

- loses absolute score magnitude
- works best as a companion plot, not the only one

#### Plot 1D: Scatter Plot of MVP vs IntPhys2

Layout:

- x-axis: `MVP pair consistency`
- y-axis: `IntPhys2 VOE`
- one point per model

Why generate it:

- useful for asking whether the two benchmarks agree
- helps show that strong performance on one benchmark does not guarantee equally strong performance on the other

Weakness:

- less intuitive as a first main figure
- best as a secondary figure or appendix figure

### Best Candidate for Main Paper

- safest: `Table 1A`
- best figure candidate: `Plot 1A`
- most interesting secondary variant: `Plot 1C`

---

## Experiment 2

### Working Title

Where is the physics signal localized in depth?

### Core Question

At what representational depth does intuitive-physics information become maximally accessible?

### Definition of Localization

In this experiment, "localization" means:

- where the performance peak occurs across depth
- whether performance rises steadily with depth or peaks earlier
- whether the optimal depth differs by benchmark

### Why This Experiment Matters

This is where the paper becomes more than a leaderboard.

It addresses whether physics-like structure is:

- early and broadly available
- late and task-ready
- or concentrated in intermediate representations

### Data Slice

- Main version: `MLP`
- Datasets: `MVP`, `IntPhys2`
- Non-LTX models:
  - use relative depth `{0.25, 0.5, 0.75, 1.0}`
- LTX:
  - use the denoising grid already available in the workbook
  - `noise level x block`

### Table Variants

#### Table 2A: Best-Layer Summary

Rows:

- one row per model

Columns:

- `Model`
- `Best layer on MVP`
- `Best MVP score`
- `Best layer on IntPhys2`
- `Best IntPhys2 score`

Why generate it:

- simple and directly supports the localization claim

#### Table 2B: Depth Profile Summary

Rows:

- one row per model

Columns:

- `Model`
- `Depth trend on MVP`
- `Depth trend on IntPhys2`
- `Peak location type`

Possible labels:

- `late-rising`
- `intermediate peak`
- `flat`
- `non-monotonic`

Why generate it:

- more interpretive
- useful for writing the discussion

#### Table 2C: LTX Best-Slot Summary

Rows:

- one row per LTX probe

Columns:

- `Probe`
- `Best noise level on MVP`
- `Best block on MVP`
- `Best MVP score`
- `Best noise level on IntPhys2`
- `Best block on IntPhys2`
- `Best IntPhys2 score`

Why generate it:

- isolates the LTX story cleanly

### Plot Variants

#### Plot 2A: Small-Multiple Line Plot by Model

Layout:

- `2 columns = 2 datasets`
- `1 row per model`

Encoding:

- x-axis: relative depth
- y-axis: primary metric
- one line per probe if we want multi-probe comparison, or just one line if we keep `MLP` fixed

Why generate it:

- most detailed version
- good for appendix or internal inspection

Weakness:

- can become too tall for the main paper

#### Plot 2B: Small-Multiple Line Plot by Family

Layout:

- `2 columns = 2 datasets`
- `1 row per family`

Encoding:

- one line per model within the family

Why generate it:

- keeps the family story visible
- much cleaner than a model-by-model grid
- strong candidate for main text

#### Plot 2C: Single Two-Panel Mean Depth Curve

Layout:

- `2 columns = 2 datasets`

Encoding:

- x-axis: relative depth
- y-axis: primary metric
- one line per family mean
- optional shaded range or thin background lines for individual models

Why generate it:

- very clean summary
- shows the broad trend without overloading the figure

Weakness:

- hides individual model exceptions

#### Plot 2D: Heatmap of Model x Depth

Layout:

- one heatmap per dataset

Encoding:

- rows: model
- columns: relative depth
- color: primary metric

Why generate it:

- compact
- very easy to scan for peak positions

Weakness:

- less expressive than line plots for showing trajectory shape

#### Plot 2E: LTX Heatmap

Layout:

- one heatmap for `MVP`
- one heatmap for `IntPhys2`

Encoding:

- x-axis: block
- y-axis: noise level
- color: primary metric

Why generate it:

- this is probably the best way to visualize LTX
- preserves the denoising structure instead of flattening it

#### Plot 2F: Peak-Only Scatter

Layout:

- x-axis: best layer position
- y-axis: best score
- one point per model

Why generate it:

- useful if we want a compact localization summary
- can show whether higher-scoring models tend to peak later

Weakness:

- loses the full curve shape

### Best Candidate for Main Paper

- likely best main figure: `Plot 2B`
- best compact alternative: `Plot 2D`
- best LTX-specific addition: `Plot 2E`

---

## Experiment 3

### Working Title

Which probe extracts the physics signal best?

### Core Question

How easily is the intuitive-physics signal readable from the frozen representation, and how much does probe expressivity matter?

### Why This Experiment Matters

This experiment gives meaning to the probe comparison.

The point is not just that one probe is numerically better. The point is to test whether the signal is:

- linearly exposed
- recoverable only with a nonlinear probe
- or better extracted with explicit temporal modeling

### Data Slice

- Datasets: `MVP`, `IntPhys2`
- Probes:
  - `linear`
  - `MLP`
  - `attentive`
- Each probe uses its own best validation layer
- Report the test score at that best layer

### Table Variants

#### Table 3A: Best Probe Per Model

Rows:

- one row per model

Columns:

- `Model`
- `Best linear score`
- `Best MLP score`
- `Best attentive score`
- `Winning probe on MVP`
- `Winning probe on IntPhys2`

Why generate it:

- direct and easy to interpret

#### Table 3B: Gain-Over-Linear Table

Rows:

- one row per model

Columns:

- `Model`
- `MLP - linear` on `MVP`
- `attentive - linear` on `MVP`
- `MLP - linear` on `IntPhys2`
- `attentive - linear` on `IntPhys2`

Why generate it:

- much more informative than raw scores
- directly quantifies accessibility beyond linear separability

#### Table 3C: Probe-Winner Count Table

Rows:

- one row per probe

Columns:

- `Number of wins on MVP`
- `Number of wins on IntPhys2`
- `Average margin over second-best`

Why generate it:

- useful for a compact summary paragraph

### Plot Variants

#### Plot 3A: Grouped Bar Chart by Model

Layout:

- `2 columns = 2 datasets`
- x-axis: model
- grouped bars: `linear`, `MLP`, `attentive`

Why generate it:

- straightforward
- easy to compare probes within each model

Weakness:

- dense if the attentive results are incomplete or if labels are crowded

#### Plot 3B: Gain-Over-Linear Bars

Layout:

- `2 columns = 2 datasets`
- x-axis: model
- bars:
  - `MLP - linear`
  - `attentive - linear`

Why generate it:

- strongest interpretation
- directly shows how much richer readout helps

#### Plot 3C: Scatter Plot Against Linear Baseline

Layout:

- left panel: `linear` vs `MLP`
- right panel: `linear` vs `attentive`

Encoding:

- x-axis: best `linear`
- y-axis: best comparison probe
- diagonal `y = x`

Why generate it:

- elegant and compact
- immediately shows whether gains are systematic

#### Plot 3D: Probe Ranking Heatmap

Layout:

- one heatmap per dataset

Encoding:

- rows: model
- columns: probe
- color: primary metric
- optionally annotate each cell with the score

Why generate it:

- compact and visually neat
- useful when there are many models

Weakness:

- weaker at showing relative gains than Plot 3B

#### Plot 3E: Dumbbell Plot Per Model

Layout:

- one panel per dataset
- one row per model
- points for `linear`, `MLP`, `attentive`
- connect the points

Why generate it:

- nice if the story is about probe improvement trajectories

### Best Candidate for Main Paper

- most interpretable: `Table 3B` plus `Plot 3B`
- most visually pleasing candidate: `Plot 3C`

---

## Experiment 4

### Working Title

What happens under temporal control experiments?

### Core Question

How much of the original task performance survives when temporal information is disrupted?

### Why This Experiment Matters

This experiment lets us say whether the models are:

- relying mostly on static cues
- relying substantially on temporal order
- or using a mixture of both

It also helps distinguish what `MVP` and `IntPhys2` are really testing.

### Data Slice

- Controls:
  - `single-frame`
  - `frame-shuffle`
- Main evaluation:
  - use original labels
- For each `model x dataset`:
  - take the best layer from the original main-task setup
  - evaluate the controls at that same layer
- Primary metric:
  - `pair consistency` for `MVP`
  - `VOE accuracy` for `IntPhys2`

### Optional Secondary Analysis

These can be generated, but should not be the main control result:

- `single-frame + all_true`
- `single-frame + all_false`
- `frame-shuffle + all_true`
- `frame-shuffle + all_false`

These are useful for interpretation and appendix discussion, especially the `frame-shuffle + all_false` case.

### Table Variants

#### Table 4A: Main Control Table

Rows:

- one row per model

Columns:

- `Best layer`
- `Main task score`
- `Frame-shuffle score`
- `Single-frame score`
- `Drop under frame-shuffle`
- `Drop under single-frame`

Why generate it:

- this is the cleanest control table
- directly supports the key claim

#### Table 4B: Retention Table

Rows:

- one row per model

Columns:

- `Main task`
- `Shuffle retention (%)`
- `Single-frame retention (%)`

Where:

- `retention = control / main`

Why generate it:

- very useful for cross-model comparison
- makes it easier to compare datasets on a common scale

#### Table 4C: Semantic-Control Appendix Table

Rows:

- one row per model or one row per control condition

Columns:

- `all_true score`
- `all_false score`
- `Original-label score`

Why generate it:

- supports the discussion of whether the transformed clips have a clean semantic label

### Plot Variants

#### Plot 4A: Connected-Point Plot

Layout:

- `2 columns = 2 datasets`
- one line per model
- three x-axis positions:
  - `Main`
  - `Frame-shuffle`
  - `Single-frame`

Why generate it:

- probably the clearest visual summary
- shows both absolute score and drop pattern

#### Plot 4B: Drop Bar Chart

Layout:

- `2 columns = 2 datasets`
- x-axis: model
- two bars per model:
  - `drop under frame-shuffle`
  - `drop under single-frame`

Why generate it:

- directly emphasizes the effect of the controls
- simple and readable

#### Plot 4C: Retention Bar Chart

Layout:

- same as Plot 4B

Encoding:

- use `retention (%)` instead of raw drop

Why generate it:

- often more interpretable across tasks with different score scales

#### Plot 4D: Heatmap of Control Damage

Layout:

- one heatmap per dataset

Encoding:

- rows: model
- columns:
  - `Main`
  - `Shuffle`
  - `Single-frame`
- or alternatively:
  - `Drop shuffle`
  - `Drop single-frame`

Why generate it:

- compact
- easy to compare many models

#### Plot 4E: Scatter of Main Score vs Retention

Layout:

- x-axis: main-task score
- y-axis: shuffle retention or single-frame retention
- one point per model

Why generate it:

- useful if we want to ask whether stronger models are also more temporally dependent

Weakness:

- this is more exploratory than headline

### Best Candidate for Main Paper

- safest table: `Table 4A`
- strongest figure candidate: `Plot 4A`
- most compact alternative: `Plot 4B`

---

## Suggested Main-Paper Combinations

### Conservative Version

- `Table 1A` for Experiment 1
- `Plot 2B` for Experiment 2
- `Plot 3B` for Experiment 3
- `Table 4A` for Experiment 4

This is the safest version if we want a clean, readable paper.

### More Visual Version

- `Plot 1A` for Experiment 1
- `Plot 2B` plus `Plot 2E` for Experiment 2
- `Plot 3C` for Experiment 3
- `Plot 4A` for Experiment 4

This version is stronger visually, but only if the plots render cleanly.

### More Compact Version

- `Table 1B`
- `Plot 2D`
- `Plot 3D`
- `Table 4B`

This version is good if the paper needs to save space.

---

## Good Appendix Items

- full per-layer tables for all `model x probe x dataset`
- model-level localization plots from `Plot 2A`
- semantic-control tables with `all_true` and `all_false`
- full `LTX` denoising heatmaps
- attentive-probe detailed comparisons if the main paper uses `MLP`

---

## Final Note

The main decision should be made after generating 2-3 visual variants for each experiment and seeing which one is:

- clearest
- least cluttered
- most visually balanced across the two datasets
- easiest to explain in one paragraph

The experiment definitions above should stay stable even if the final plot choice changes.
