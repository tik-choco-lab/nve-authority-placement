Medoid control evaluated with the original replay engine on all 420 original ambiguity traces (20 peers, 5 entities, lambda=2 requests/entity/s, tau=10 s, 300 s duration, alpha=1). Each entity's first request is served at its original initial holder; the policy then returns the RTT-matrix medoid, with peer-ID tie breaking, and stays there. It reads no request history or future requests. A holder already at the medoid incurs no migration.

Weighted and Argmax settings minimize the committed mean combined cost separately at each ratio over the full original grid (w in {5,10,20,30,100}; theta_D in {0,0.25,0.5,1,2,5,10} s). The better Static/Nearest baseline is Nearest at every ratio. All choices use seeds 1–30 only and are frozen for seeds 31–60; exact ties retain committed CSV order. Both follower policies use theta_G=0, theta_R=1e9, count windows, and unweighted interactions.

Frozen (w, theta_D in seconds), in ratio order 0.85 / 0.70 / 0.60 / 0.50 / 0.40 / 0.30 / 0.25: Weighted = (5, 0) / (10, 0) / (10, 0.25) / (10, 0.25) / (100, 0.25) / (100, 0.25) / (100, 1); Argmax = (5, 0) / (5, 0) / (10, 0) / (10, 0) / (10, 0.25) / (10, 1) / (5, 5).

Each cost is a mean across 30 seeds of [sum of served interaction latencies + sum of migration RTTs]/1000, including the original 2 ms processing delay per request. Costs and differences below are seconds. M=Medoid, W=tuned Weighted, A=tuned Argmax, B=frozen better baseline (Nearest). S=selection seeds 1–30; H=held-out seeds 31–60. Brackets are seed-paired percentile 95% bootstrap CIs, using the unchanged original `experiments.load_significance.paired_bootstrap` function with 10,000 resamples and RNG seed 12345. Negative differences favor Medoid. Imbalance is computed per seed as max served requests / mean over all 20 peers, then averaged; 1 is even and 20 is the maximum. CIs are pointwise per comparison.

| Ratio | Split | M cost | W cost | A cost | B cost | M − W [95% CI] | M − B [95% CI] | Imbalance M / W |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | S | 236.33 | 117.97 | 114.67 | 302.18 | +118.37 [+111.02, +125.73] | -65.84 [-75.23, -56.68] | 19.97 / 1.83 |
| 0.85 | H | 244.37 | 117.14 | 113.25 | 306.83 | +127.22 [+120.71, +133.03] | -62.47 [-70.30, -55.06] | 19.97 / 1.69 |
| 0.70 | S | 236.65 | 178.70 | 171.53 | 302.07 | +57.94 [+51.11, +64.76] | -65.43 [-74.45, -56.66] | 19.97 / 2.57 |
| 0.70 | H | 245.06 | 180.07 | 170.76 | 307.28 | +64.99 [+59.57, +69.92] | -62.22 [-69.90, -54.95] | 19.97 / 2.32 |
| 0.60 | S | 237.19 | 210.01 | 206.51 | 302.73 | +27.17 [+21.44, +32.90] | -65.55 [-75.03, -56.43] | 19.97 / 3.11 |
| 0.60 | H | 244.54 | 212.14 | 205.24 | 306.76 | +32.39 [+27.62, +36.55] | -62.22 [-70.27, -54.72] | 19.97 / 2.87 |
| 0.50 | S | 236.51 | 233.45 | 238.91 | 302.53 | +3.05 [-1.83, +8.11] | -66.02 [-75.37, -56.94] | 19.97 / 3.58 |
| 0.50 | H | 244.19 | 235.78 | 237.47 | 307.40 | +8.41 [+4.55, +11.65] | -63.21 [-71.48, -55.47] | 19.97 / 3.26 |
| 0.40 | S | 237.29 | 249.88 | 270.79 | 302.20 | -12.60 [-15.35, -9.93] | -64.91 [-74.04, -56.13] | 19.97 / 8.67 |
| 0.40 | H | 244.91 | 258.40 | 270.30 | 307.26 | -13.49 [-15.66, -11.31] | -62.35 [-70.27, -55.07] | 19.97 / 7.42 |
| 0.30 | S | 236.48 | 251.63 | 300.22 | 303.25 | -15.15 [-17.48, -12.74] | -66.77 [-75.72, -58.10] | 19.97 / 9.64 |
| 0.30 | H | 245.01 | 257.72 | 302.42 | 306.84 | -12.71 [-14.49, -10.98] | -61.83 [-69.78, -54.45] | 19.97 / 8.60 |
| 0.25 | S | 236.96 | 250.15 | 304.59 | 302.97 | -13.20 [-15.27, -11.14] | -66.01 [-75.02, -57.27] | 19.97 / 10.43 |
| 0.25 | H | 245.62 | 259.52 | 305.56 | 307.63 | -13.90 [-15.42, -12.37] | -62.01 [-70.11, -54.37] | 19.97 / 8.99 |

Validation passed: 1,232 numeric checks on seeds 1–30; maximum absolute difference from committed values was 4.99334675e-07 (tolerance 5.1e-07, allowing six-decimal rounding). Static and selected Weighted reproduce `weighted_dwell/aggregate.csv` at all seven ratios for mean and standard deviation of combined cost, mean latency, and mean migration count. Selected Argmax and Nearest also pass those checks. Static additionally reproduces `ambiguity/sweep_aggregate.csv` for mean/std latency, p95 latency, migrations, maximum requests served, and imbalance, plus all 210 committed Static seed records for those five metrics. Every Medoid replay passes a service-order and migration-charge audit, including the original initial holder on first service and at most one move per entity; an unsorted-peer tie test also passes.

Medoid's mean cost changes little with ambiguity: 236.3–237.3 s on selection seeds and 244.2–245.6 s on held-out seeds. On held-out seeds, Medoid costs 8.4–127.2 s more than tuned Weighted at ratios 0.50–0.85 and 12.7–13.9 s less at ratios 0.25–0.40; all seven paired intervals exclude zero. Selection seeds show the same directions, but the interval at ratio 0.50 includes zero. Medoid costs less than the frozen better baseline at every ratio in both splits, with all paired intervals excluding zero. Its processing imbalance is about 19.97 out of a maximum 20, compared with 1.69–10.43 for tuned Weighted across these runs: almost all Medoid service occurs at a single peer. Thus the cost advantage over tuned Weighted at the three lowest ratios accompanies substantially greater processing concentration.

Reproduce with `uv run python -m experiments.medoid_control`. It uses two worker processes and does not modify existing datasets. `medoid_results.csv` contains all 14 rows at six decimals; `medoid_per_seed.csv`, `frozen_selection.json`, and `validation.json` provide the seed values, frozen settings, and checks.
