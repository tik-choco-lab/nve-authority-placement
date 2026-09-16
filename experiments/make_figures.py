"""Render the paper's central figures.

fig_switching.pdf reads
datasets/switching/sweep_aggregate.csv (experiments/switching_sweep.py).
fig_dwell_tradeoff.pdf - the key figure - reads
datasets/ambiguity/sweep_aggregate.csv (experiments/ambiguity_sweep.py)
instead: the switching sweep found that at its dominant_peer_ratio (0.85)
there is almost no transient target churn for theta_D to filter (see
ambiguity_sweep.py's module docstring), so the dwell/latency tradeoff is
shown against the ambiguity axis (dominant_peer_ratio) where that churn
actually exists. fig_processing_load.pdf also reads the ambiguity aggregate:
it answers whether the interaction-following family concentrates processing
load (LB-Spiral's second objective, alongside stretch) more than the
baselines, and whether theta_D's migration suppression makes that
concentration worse by pinning authority in place for longer.
fig_boundary_collapse.pdf reads datasets/boundary/collapse.csv
(experiments/boundary_collapse.py): it plots the seed-paired margin between
the best interaction-following policy and the better baseline, at three
request rates, against the measured dimensionless axis rho_eff instead of
the request-rate-specific dominant peer ratio, to show that the "boundary
near ratio=0.30" the other figures live at is a single rho_eff~=1 crossing
in disguise, not three different rate-specific boundaries.
Every figure reads whichever CSV it needs directly, so this script can be
re-run (e.g. after tweaking a label or a color) without redoing any sweep.
All output is vector PDF, sized for an IEEE two-column single-column figure
slot.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
SWITCHING_AGGREGATE_CSV = ROOT / "datasets" / "switching" / "sweep_aggregate.csv"
AMBIGUITY_AGGREGATE_CSV = ROOT / "datasets" / "ambiguity" / "sweep_aggregate.csv"
BOUNDARY_COLLAPSE_CSV = ROOT / "datasets" / "boundary" / "collapse.csv"
# Keep generated figures in this repository; NVE_FIGURES_DIR overrides it.
FIGURES_DIR = Path(os.environ.get("NVE_FIGURES_DIR", ROOT / "figures"))

FIXED_PHASE_DURATION = 10.0  # the switching interval fig_switching_*.pdf hold constant
DWELL_ID_RE = re.compile(r"^engagement_g(\d\d)_d([0-9.]+)$")

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def load_aggregate(path: Path, group_column: str) -> dict[tuple[float, str], dict[str, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (float(row[group_column]), row["policy_id"])
            rows[key] = {name: float(row[name]) for name in row if name not in (group_column, "policy_id")}
    return rows


def phase_durations(rows: dict[tuple[float, str], dict[str, float]]) -> list[float]:
    return sorted({phase for phase, _ in rows})


def log_x_axis(ax: plt.Axes, phases: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(phases)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("target switching interval (s)")


def fig_switching(rows: dict[tuple[float, str], dict[str, float]]) -> None:
    """Latency and migration count against the switching interval, stacked.

    Both panels share the switching-interval axis.
    """
    phases = phase_durations(rows)
    series = [
        ("static", "Static", {}),
        ("nearest", "Nearest", {}),
        ("naive_target_aware", "Naive Target-Aware", {}),
        ("engagement_g00_d0.25", r"Proposed ($\theta_D{=}0.25$ s)", {}),
        ("oracle_a1", r"Oracle ($\alpha{=}1$)", {"linestyle": "--"}),
    ]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True,
                                      constrained_layout=True)
    for policy_id, label, style in series:
        means = [rows[(phase, policy_id)]["mean_latency_ms_mean"] for phase in phases]
        stds = [rows[(phase, policy_id)]["mean_latency_ms_std"] for phase in phases]
        top.errorbar(phases, means, yerr=stds, marker="o", markersize=3, capsize=2,
                     linewidth=1, label=label, **style)
        # The Oracle's migration count is not a meaningful baseline on this axis:
        # it buys its latency with migrations no online policy would accept.
        if policy_id == "oracle_a1":
            continue
        means = [rows[(phase, policy_id)]["migration_count_mean"] for phase in phases]
        stds = [rows[(phase, policy_id)]["migration_count_std"] for phase in phases]
        bottom.errorbar(phases, means, yerr=stds, marker="o", markersize=3, capsize=2,
                        linewidth=1, label=label, **style)

    top.set_ylabel("mean latency (ms)")
    # y stays linear: Static is exactly 0 migrations at every interval, which a log
    # axis cannot represent at all.
    bottom.set_ylabel("migrations / 300 s")
    for ax in (top, bottom):
        ax.grid(True, alpha=0.3)
    log_x_axis(bottom, phases)

    # Below the axes, not inside: every in-axes corner is occupied here - the flat
    # baselines sit across the top and the decaying curves sweep the diagonal - so
    # an inset legend would cover data at some interval no matter where it is put.
    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2,
               frameon=False, fontsize=6, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(FIGURES_DIR / "fig_switching.pdf", bbox_inches="tight")
    plt.close(fig)


# fig_dwell_tradeoff draws one curve per dominant_peer_ratio, at engagement_threshold=0.0
# only (theta_G=0.5 is switching_sweep.py's territory; this figure is about the
# ambiguity axis, not the engagement-threshold axis - see the module docstring).
# 0.60 is dropped: it overlaps 0.40 almost everywhere on these axes, is never
# quoted in the paper, and the three kept ratios already bracket the argument.
DWELL_RATIOS = [0.85, 0.40, 0.25]
DWELL_ANNOTATE_RATIO = 0.25  # the curve that gets per-theta_D text labels
# theta_D label positions, in data coordinates, with a leader line drawn back to
# the marker: an offset alone left every label ambiguous, because the curves
# converge near latency 100ms and a label parked in that tangle reads as
# belonging to whichever series happens to sit under it. Labelling every swept
# theta_D was tried first and reported unreadable (six small, crowded labels);
# the direction of traversal is already carried by the "theta_D increases"
# arrow below, so the labelled set is trimmed to the four values the text and
# Table I actually quote (0, 0.25, 1, 2) - 0.5, 5, and 10 keep their markers
# and error bars but lose their text. All four sit in the headroom above their
# own marker, staggered in height (not a single row) because the two nearest
# markers in x (0.25 and 1.0's neighbors) are less than a quarter-decade apart
# on this symlog axis - close enough that same-height labels would collide.
DWELL_LABEL_POS = {
    2.0: (200.0, 100.0),
    1.0: (330.0, 112.0),
    0.25: (460.0, 124.0),
    0.0: (610.0, 138.0),
}


def _dwell_curve(rows: dict[tuple[float, str], dict[str, float]], ratio: float) -> list[tuple[float, float, float, float, float]]:
    """Returns (dwell, migration_mean, migration_std, latency_mean, latency_std) sorted by
    dwell, for the engagement_threshold=0.0 arm at one dominant_peer_ratio."""
    points = []
    for (row_ratio, policy_id), values in rows.items():
        if row_ratio != ratio:
            continue
        match = DWELL_ID_RE.match(policy_id)
        if match is None or match.group(1) != "00":
            continue
        dwell = float(match.group(2))
        points.append((dwell, values["migration_count_mean"], values["migration_count_std"],
                        values["mean_latency_ms_mean"], values["mean_latency_ms_std"]))
    return sorted(points)


def fig_dwell_tradeoff(rows: dict[tuple[float, str], dict[str, float]]) -> None:
    """The paper's key figure. Reads datasets/ambiguity/sweep_aggregate.csv (NOT the
    switching aggregate - see module docstring): dominant_peer_ratio, not
    phase_duration, is the axis that actually produces transient target churn for
    theta_D to filter, so this is where the dwell/latency tradeoff is worth plotting.
    """
    # Height, not width, is the scarce resource: the paper scales this to
    # \columnwidth, so the column height it eats is set by the saved aspect ratio
    # (bbox_inches="tight" trims to content, and a long xlabel would widen the box
    # and shrink the plot to compensate). 2.1in keeps it under one third column.
    fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)

    colors = {0.85: "C0", 0.40: "C2", 0.25: "C3"}
    markers = {0.85: "o", 0.40: "^", 0.25: "D"}

    for ratio in DWELL_RATIOS:
        points = _dwell_curve(rows, ratio)
        dwells = [p[0] for p in points]
        mig = [p[1] for p in points]
        mig_err = [p[2] for p in points]
        lat = [p[3] for p in points]
        lat_err = [p[4] for p in points]
        color = colors[ratio]
        ax.errorbar(mig, lat, xerr=mig_err, yerr=lat_err, marker=markers[ratio], markersize=4,
                    linewidth=1, capsize=2, color=color, label=f"ratio={ratio:g}")
        # Annotate only the ratio=0.25 curve: it has the most transient churn (see
        # ambiguity_sweep.py's migration_floor gap) and stacking theta_D labels on
        # all three curves would be unreadable at this figure size.
        if ratio != DWELL_ANNOTATE_RATIO:
            continue
        for dwell, mx, my in zip(dwells, mig, lat):
            if dwell in DWELL_LABEL_POS:
                ax.annotate(rf"$\theta_D{{=}}{dwell:g}$", xy=(mx, my), xycoords="data",
                            xytext=DWELL_LABEL_POS[dwell], textcoords="data",
                            fontsize=8, color=color, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                      edgecolor="none", alpha=0.75),
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=color,
                                            shrinkA=1, shrinkB=3))

    baselines = [
        ("static", "Static", "*", "black"),
        ("nearest", "Nearest", "x", "dimgray"),
    ]
    for policy_id, label, marker, color in baselines:
        values = rows[(DWELL_ANNOTATE_RATIO, policy_id)]
        ax.scatter(values["migration_count_mean"], values["mean_latency_ms_mean"],
                   marker=marker, s=32, color=color,
                   label=f"{label} (ratio={DWELL_ANNOTATE_RATIO:g})", zorder=5)

    # symlog, not plain log: Static has exactly 0 migrations, which a pure log axis
    # cannot represent at all, and the migration range above that spans close to
    # two orders of magnitude (roughly 9 at high theta_D / low ratio up to ~690 at
    # theta_D=0, ratio=0.25) - wide enough that linear would crush the low end.
    # symlog is linear below linthresh and log above it, so 0 stays representable
    # while the wide upper range still gets log compression.
    ax.set_xscale("symlog", linthresh=10)
    # Headroom above the curves for the right-hand theta_D label stack.
    ax.set_ylim(38, 120)
    ax.set_xlabel("authority migrations per 300\u2009s run")
    ax.set_ylabel("mean interaction latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.annotate("better", xy=(0.05, 0.05), xycoords="axes fraction",
                xytext=(0.30, 0.22), textcoords="axes fraction",
                fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.8))
    # theta_D runs right to left on every curve, which no reader recovers from the
    # labels alone at a glance.
    ax.annotate("", xy=(230.0, 58.0), xytext=(640.0, 58.0), xycoords="data",
                textcoords="data", arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
    ax.text(380.0, 59.5, r"$\theta_D$ increases", fontsize=6, color="0.35", ha="center",
            va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              frameon=False, fontsize=6, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(FIGURES_DIR / "fig_dwell_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


# fig_processing_load answers the processing-load question directly: request_imbalance
# against theta_D, at engagement_threshold=0.0 (same arm as fig_dwell_tradeoff), for the two
# ratios that bracket the ambiguity sweep. Two curves only (not all DWELL_RATIOS): more
# would clutter a 3.4in-wide axis, and the two extremes already show whether theta_D's
# direction of effect depends on ambiguity.
LOAD_RATIOS = [0.85, 0.25]
LOAD_BASELINE_RATIO = 0.25  # ratio at which the flat static/nearest references are drawn


def _load_curve(rows: dict[tuple[float, str], dict[str, float]], ratio: float) -> list[tuple[float, float, float]]:
    """Returns (dwell, request_imbalance_mean, request_imbalance_std) sorted by dwell,
    for the engagement_threshold=0.0 arm at one dominant_peer_ratio."""
    points = []
    for (row_ratio, policy_id), values in rows.items():
        if row_ratio != ratio:
            continue
        match = DWELL_ID_RE.match(policy_id)
        if match is None or match.group(1) != "00":
            continue
        dwell = float(match.group(2))
        points.append((dwell, values["request_imbalance_mean"], values["request_imbalance_std"]))
    return sorted(points)


def fig_processing_load(rows: dict[tuple[float, str], dict[str, float]]) -> None:
    """Does pushing authority toward the peer interacting most with an entity actually
    concentrate processing load on that peer (LB-Spiral's second objective), and does
    theta_D make it better or worse? x is theta_D on a LINEAR axis, not log: unlike
    the switching-interval axis (which spans two decades), DWELL_VALUES only runs
    0..10 and includes 0.0 itself, which a log axis cannot place.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)

    colors = {0.85: "C0", 0.25: "C3"}
    markers = {0.85: "o", 0.25: "D"}
    for ratio in LOAD_RATIOS:
        points = _load_curve(rows, ratio)
        dwells = [p[0] for p in points]
        means = [p[1] for p in points]
        stds = [p[2] for p in points]
        ax.errorbar(dwells, means, yerr=stds, marker=markers[ratio], markersize=4,
                    linewidth=1, capsize=2, color=colors[ratio],
                    label=rf"Proposed, ratio={ratio:g} ($\theta_G{{=}}0$)")

    baselines = [
        ("static", "Static", "black", "--"),
        ("nearest", "Nearest", "dimgray", ":"),
    ]
    for policy_id, label, color, linestyle in baselines:
        value = rows[(LOAD_BASELINE_RATIO, policy_id)]["request_imbalance_mean"]
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=1,
                   label=f"{label} (ratio={LOAD_BASELINE_RATIO:g})")

    ax.set_xlabel(r"dwell threshold $\theta_D$ (s)")
    ax.set_ylabel("request imbalance (max / mean per peer)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              frameon=False, fontsize=6, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(FIGURES_DIR / "fig_processing_load.pdf", bbox_inches="tight")
    plt.close(fig)


def load_collapse_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({name: (value if name in ("best_follower", "best_baseline", "verdict") else float(value))
                         for name, value in row.items()})
    return rows


# Plot measured self-service benefit and migration cost across request rates.
# These are outcome-derived explanatory measures.
COLLAPSE_RATE_STYLE = {0.5: ("C0", "o"), 2.0: ("C2", "^"), 8.0: ("C3", "D")}

# One measured point (rate=0.5, ratio=0.25) has rho_eff = -0.0032: s is
# indistinguishable from s0 within seed noise there (see
# datasets/boundary/collapse.csv), so the axis is a hair negative rather than
# exactly 0. A log x axis cannot place a negative value at all, and every
# other sampled rho_eff is >= 0.017, so this single point is floored to
# RHO_EFF_FLOOR for its x POSITION only (its CSV value, and the plotted
# curve's shape everywhere else, are untouched) rather than switching the
# whole axis to symlog for one point that is itself consistent with zero.
RHO_EFF_FLOOR = 0.01


def fig_boundary_collapse(rows: list[dict[str, float]]) -> None:
    # 3.4 x 1.9in, not 3.4 x 2.5in: the paper is over its 6-page limit and
    # column height is the binding constraint, so the legend moves inside the
    # empty upper-right quadrant (large rho_eff and a positive margin never
    # co-occur in this data - see the plot) instead of costing a strip below
    # the axes, and the aspect ratio is trimmed to roughly match
    # fig_dwell_tradeoff's column cost instead of fig_processing_load's.
    fig, ax = plt.subplots(figsize=(3.4, 1.5), constrained_layout=True)

    for rate in sorted(COLLAPSE_RATE_STYLE):
        points = sorted((r for r in rows if r["rate_per_entity"] == rate), key=lambda r: r["rho_eff"])
        if not points:
            continue
        x = [max(p["rho_eff"], RHO_EFF_FLOOR) for p in points]
        y = [p["mean_diff_combined_s"] for p in points]
        yerr_lo = [p["mean_diff_combined_s"] - p["bootstrap_ci95_low"] for p in points]
        yerr_hi = [p["bootstrap_ci95_high"] - p["mean_diff_combined_s"] for p in points]
        color, marker = COLLAPSE_RATE_STYLE[rate]
        ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], marker=marker, markersize=4, linewidth=1,
                    capsize=2, color=color, linestyle="-", label=rf"$\lambda={rate:g}$ req/s/entity")

    ax.set_xscale("log")
    # symlog, not plain log, on y only: the margin changes sign at the crossing
    # (see fig_dwell_tradeoff's identical reasoning for why a sign change rules
    # out a pure log axis), and the sampled margins span close to three orders
    # of magnitude on either side of it (3.7 s to 899 s). Minor-tick labels are
    # suppressed on both axes (as in log_x_axis above) - left on, the symlog
    # minor ticks flanking zero collide with the "0" and "10^0" major labels at
    # this figure's width.
    ax.set_yscale("symlog", linthresh=5.0, linscale=0.6)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Explicit major ticks: symlog's default locator also places +-1 (inside
    # linthresh) right next to the 0 tick, which at this figure's width prints
    # "1" and "-1" on top of "0". No sampled margin falls in (-5, 5) except the
    # crossing itself, so those two ticks add clutter without adding information.
    ax.set_yticks([-1000, -100, -10, 0, 10])
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.scatter([1.0], [0.0], marker="*", s=70, color="black", zorder=5,
               label=r"predicted: $\rho_{\mathrm{eff}}{=}1$")

    ax.set_xlabel(r"$\rho_{\mathrm{eff}} = (s-s_0)\,\lambda \tau / (\alpha m)$")
    ax.set_ylabel("seed-paired margin,\nfollowing $-$ better baseline (s)")
    ax.grid(True, alpha=0.3)
    # Inside the axes, not below: the direction of the y axis is already named
    # in its label ("following - better baseline"), so the legend only needs
    # to say which marker is which rate. upper right is empty in this data -
    # large rho_eff only ever co-occurs with a large NEGATIVE margin - so the
    # legend sits there instead of costing a strip of column height below the
    # axes, unlike this script's other legends.
    ax.legend(loc="upper right", fontsize=6, frameon=True, framealpha=0.85,
              edgecolor="none", columnspacing=1.0, handletextpad=0.4, borderpad=0.4)
    fig.savefig(FIGURES_DIR / "fig_boundary_collapse.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    switching_rows = load_aggregate(SWITCHING_AGGREGATE_CSV, "phase_duration")
    fig_switching(switching_rows)
    print(f"[write] {FIGURES_DIR / 'fig_switching.pdf'}")

    ambiguity_rows = load_aggregate(AMBIGUITY_AGGREGATE_CSV, "dominant_peer_ratio")
    fig_dwell_tradeoff(ambiguity_rows)
    print(f"[write] {FIGURES_DIR / 'fig_dwell_tradeoff.pdf'}")

    fig_processing_load(ambiguity_rows)
    print(f"[write] {FIGURES_DIR / 'fig_processing_load.pdf'}")

    if BOUNDARY_COLLAPSE_CSV.is_file():
        collapse_rows = load_collapse_rows(BOUNDARY_COLLAPSE_CSV)
        fig_boundary_collapse(collapse_rows)
        print(f"[write] {FIGURES_DIR / 'fig_boundary_collapse.pdf'}")
    else:
        print(f"[skip]  {BOUNDARY_COLLAPSE_CSV.relative_to(ROOT)} not found - "
              "run experiments/boundary_collapse.py first")


if __name__ == "__main__":
    main()
