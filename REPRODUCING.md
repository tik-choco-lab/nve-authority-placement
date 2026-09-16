# Reproducing the paper

This repository contains the workload generator, replay harness and result
CSVs for synthetic authority-placement experiments. The table below maps
analyses to commands. Exact correspondence to a particular manuscript version
requires checking that version against these outputs.

This file is the artifact contract: what to install, what to run, and which
command emits which number. `README.md` introduces the project in English;
`README.ja.md` provides a Japanese introduction. This
file is only about re-deriving the results.

## 1. Environment

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Nothing else: the
runtime dependencies are `numpy`, `matplotlib` and `PyYAML`, pinned in
`uv.lock`. There is no scipy, so every confidence interval here is a paired
bootstrap or a Student's-t interval implemented in
`experiments/load_significance.py`.

```bash
uv sync --locked # installs the locked dependency set
uv run pytest
```

## 2. The short path: nothing to regenerate

The aggregated result CSVs the paper cites are committed. If you only want to
check that the numbers in the paper follow from the data, no sweep needs to be
re-run:

```bash
uv run python -m experiments.paper_numbers          # Section V-D margins and stretch
uv run python -m experiments.theta_d_global_optimum # is theta_D > 0 ever globally optimal
uv run python -m experiments.ambiguity_30seed_report
uv run python -m experiments.make_figures           # four vector PDFs; only
                                                    # fig_boundary_collapse.pdf
                                                    # is included in the paper
```

Each takes a couple of seconds. `make_figures` writes into `figures/`
inside this repository; set `NVE_FIGURES_DIR` to send them elsewhere.

## 3. Which command produces which result

| Paper | Result | Regenerate the data with | Read the numbers with |
|---|---|---|---|
| §V-B | Latency and migrations vs. target switching interval (figure generated, not included in the 6-page paper) | `experiments.switching_sweep` | `experiments.make_figures` |
| §V-C | Dwell time / migration trade-off vs. target ambiguity (figure generated, not included) | `experiments.ambiguity_sweep` | `experiments.make_figures` |
| §V-F | Processing-load imbalance (figure generated, not included in the 6-page paper) | `experiments.ambiguity_sweep` | `experiments.make_figures` |
| Table I | Combined objective, all policies x ratios x alpha | `experiments.combined_objective` | `experiments.theta_d_global_optimum` |
| §V-B | 0.25 s dwell is nearly free on the switching axis | `experiments.switching_sweep` | `datasets/switching/sweep_aggregate.csv` |
| §V-C | The engagement threshold produces no undominated point | `experiments.ambiguity_sweep` | `experiments.ambiguity_30seed_report` |
| §V-D | Seed-paired margins around theta_D = 0.25 s at ratio 0.40 | `experiments.combined_objective` | `experiments.paper_numbers` |
| §V-D | The boundary moves with the request rate (0.5 / 2 / 8 req/s) | `experiments.boundary_sensitivity` | `datasets/boundary/rate_comparisons.csv` |
| §V-D | Lowest RTT is cheaper than every baseline at every ratio | `experiments.rtt_baseline` | `datasets/rtt_baseline/comparisons.csv` |
| §V-D | Stretch factors 2.38 / 2.21 / 2.02 (Table I's cheapest online policy over the Oracle) | `experiments.weighted_dwell`, `experiments.combined_objective` | `datasets/weighted_dwell/aggregate.csv`, `datasets/combined_objective/aggregate.csv` |
| -- | Stretch over the narrower `rtt_baseline` policy set (2.69 / 2.24 / 2.08); not what the paper quotes | `experiments.rtt_baseline` | `datasets/rtt_baseline/stretch.csv` |
| Table I, §V-D | Both rules over the 5x7 window/dwell grid: cell costs, best cell per ratio, margins and CIs | `experiments.weighted_dwell` | `datasets/weighted_dwell/{aggregate,comparisons}.csv` |
| Fig. 1, §V-D | rho and rho_eff per (rate, ratio), and where the margin changes sign | `experiments.boundary_collapse` | `datasets/boundary/collapse.csv` |
| §V-D | Held-out confirmation: the selected cells re-evaluated on seeds 31-60 | `experiments.holdout` | `datasets/holdout/{ratio,rate}_confirmation.csv` |
| §V-F | Processing-load imbalance of the RTT-weighted rule | `experiments.weighted_load` | `datasets/weighted_load/*.csv` |
| §V-E | Self-service explains the latency gain | `experiments.self_service` | `datasets/*/self_service_aggregate.csv` |
| §V-E | Lowest RTT moves the remote-RTT term instead | `experiments.rtt_baseline` | `datasets/rtt_baseline/aggregate.csv` |
| §V-E | Oracle solves a different problem, not the same one better | `experiments.oracle_analysis` | `datasets/ambiguity/oracle_analysis_aggregate.csv` |
| §V-F | Processing-load ordering is not noise | `experiments.load_significance` | `datasets/load_significance/*.csv` |
| §V-F | One shared dominant peer reverses the load ordering | `experiments.shared_dominant` | `datasets/shared_dominant/*.csv` |
| §V-G | theta_R needs interaction and space to be correlated | `experiments.range_sweep` | `datasets/range/sweep_aggregate.csv` |
| §V-G | The demand window matters more than theta_D, but not for the boundary | `experiments.boundary_sensitivity` | `datasets/boundary/window_aggregate.csv` |

Run any of them as `uv run python -m experiments.<name>`. Scripts in the third
column generate datasets and replay policies; scripts in the fourth column use saved results. Some write report CSVs or
figure files; they do not all run without filesystem changes.

Dependency order, if you regenerate everything: `switching_sweep` and
`ambiguity_sweep` first, then `combined_objective`, `self_service`,
`oracle_analysis`, `load_significance` and `rtt_baseline` (which read what those
two wrote), then `make_figures`. `range_sweep`, `shared_dominant` and the rate
arm of `boundary_sensitivity` generate their own datasets and are independent of
all of them; the window arm of `boundary_sensitivity` replays
`datasets/ambiguity`. `weighted_dwell` and `weighted_load` replay
`datasets/ambiguity`; `boundary_collapse` replays `datasets/boundary/rate_*` and
reads `rate_comparisons.csv`. Run `holdout` last: it reads the selections
recorded in `datasets/weighted_dwell/aggregate.csv` and
`datasets/boundary/collapse.csv`, then generates the seed 31-60 traces it needs
for both.

## 4. Seeds, sizes and runtimes

Measured on one WSL2 laptop core; all runs are single-threaded.

| Sweep | Configurations | Wall clock | Raw datasets on disk |
|---|---|---|---|
| `switching_sweep` | 6 phase durations x 5 seeds x 18 policies | 20 s clean, 15 s warm | 13 MB |
| `ambiguity_sweep` | 7 ratios x 30 seeds x 18 policies | 2 min warm | 85 MB |
| `combined_objective` | replays both sweeps at alpha in {0.5, 1, 2} | 2 min 18 s warm | reuses the above |
| `rtt_baseline` | 7 ratios x 30 seeds x 10 policies | 1 min 20 s warm | reuses `datasets/ambiguity` |
| `boundary_sensitivity` | 3 rates x 4 ratios x 30 seeds, then a 4x4 window/dwell grid | 4 min clean | 120 MB |
| `shared_dominant` | 2 assignments x 30 seeds x 4 policies | 20 s clean | 4 MB |
| `weighted_dwell` | 7 ratios x 30 seeds x 72 policies | 5 min warm | reuses `datasets/ambiguity` |
| `boundary_collapse` | 3 rates x 4 ratios x 30 seeds x 6 policies | 1 min warm | reuses `datasets/boundary` |
| `holdout` | 7 ratios + 12 (rate, ratio) cells x 30 held-out seeds, selected policies only | 1 min warm, 4 min clean | +84 MB ambiguity, +231 MB boundary |
| saved-data reports | -- | ~1-2 s each | some write derived CSVs/PDFs |

"Warm" means the raw datasets were already on disk, so the sweep skipped
generation and only replayed the policies; "clean" means it generated them too.
Only `switching_sweep` was timed both ways, and generation accounted for about
4 of its 20 seconds.

Seeds are the integers 1..30 for the ambiguity sweep and 1..5 for the switching
sweep, fixed in the scripts, not sampled. Statistical seeds are fixed too: the
paired bootstrap uses 10,000 resamples with `seed=12345`.

Seeds 1..30 are the *selection* set: every "best (window, theta_D)" in the paper
is the cheapest cell on those seeds, and the CIs the sweeps write beside it are
computed on the same seeds, so they are exploratory - they do not cover the
uncertainty of taking a minimum over 35 (or 4) candidates. `experiments.holdout`
supplies the confirmatory half: it re-reads those selections, generates seeds
31..60, and re-evaluates only the selected configurations there. Use held-out intervals for confirmatory comparisons and retain the selection-set
qualification when reporting exploratory results.

To verify deterministic regeneration, generate traces into a clean copy of
this repository, rerun the sweeps in dependency order, and compare result CSVs
against the distributed files. PDF creation timestamps may differ. Running
the saved-data reports alone does not verify this property.

## 5. What is committed and what is not

Committed: all source, `uv.lock`, the aggregated result CSVs the paper cites
(about 5 MB, `datasets/holdout/` included), and the small 12-dataset
demonstration suite under `datasets/small/`.

Not committed, because it is large and fully regenerable from a seed: the raw
per-request replay output and the sweep datasets themselves
(`datasets/ambiguity/ratio_*`, including the seed 31-60 traces `holdout`
generates, `datasets/switching/phase_*`,
`datasets/range/{coupled,uncoupled}`, `datasets/boundary/rate_*`,
`datasets/shared_dominant/{distinct,shared}`, and every `output/*/results.csv`). See
`.gitignore` for the exact list and the command that rebuilds each.

## 6. Naming: the paper's symbols against the code's identifiers

The code uses the paper's terminology throughout.

| Paper | Code |
|---|---|
| engagement $G(E,P,t)$ | `engagement` (a spec key, a variable, and a `decisions.csv` column) |
| engagement threshold $\theta_G$ | `engagement_threshold` |
| dwell time $\theta_D$ | `dwell_time_s` |
| relevant range $\theta_R$ | `relevant_range` |
| the proposed policy | `EngagementAwarePolicy`, `nve_policy/policies/engagement_aware.py` |

Policy ids in the sweeps encode the two swept knobs: `engagement_g00_d0.25` is
$\theta_G = 0$ with $\theta_D = 0.25$ s, and `engagement_g05_d1.0` is
$\theta_G = 0.5$ with $\theta_D = 1$ s. `naive_target_aware` is provably
identical to `engagement_g00_d0.0` (checked at every sweep run by
`experiments/common.py::check_equivalence`). The four ids in `policies.yaml`
also carry the range, as in `engagement_aware_r500_g05_d2`.

Section V-C of the paper reports $\theta_G$ as a condition that was tested and
discarded, so the policy it defines carries only $\theta_D$ and $\theta_R$. The
threshold stays in the code because the sweeps still measure it: that is where
the negative result comes from.

## 7. Scope

Single-authority / single-writer, fixed RTT drawn independently of position, 2D
space, static or random-walk mobility, synthetic request sequences. No cheating,
Byzantine faults, consensus, churn, packet loss, partitions, rollback, or human
behaviour model. Section V-G of the paper states the consequences of these
choices for what the results support.

## 8. License

Code is MIT (`LICENSE`); generated datasets are CC BY 4.0
(`datasets/LICENSE`).

## Report scope and standalone layout

Run commands from this repository root. `paper_numbers` reports stretch over
its fixed policy set; it does not select over the full window/dwell grid in
`weighted_dwell`. These stretch values must not be substituted for Table I's
full-grid results. `ambiguity_30seed_report` writes descriptive CSV tables;
it does not check manuscript quotations.

Only aggregate sweep CSVs and the small demonstration inputs are distributed.
Generate policy outputs with the Quick start commands. The concentrated
load-significance inputs are regenerated by `experiments.load_significance`.
A successful report run from saved CSVs is not a clean rerun of all experiments.
