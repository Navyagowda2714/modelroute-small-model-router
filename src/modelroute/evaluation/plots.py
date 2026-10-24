"""Static matplotlib charts for docs/ (cost-quality Pareto, threshold sweep, reliability)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np

from .metrics import StrategyResult

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)


def pareto_chart(sweep: list[StrategyResult], small: StrategyResult, large: StrategyResult,
                 chosen: StrategyResult, default: StrategyResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    _style(ax)
    xs = [p.cost_per_1k_usd for p in sweep]
    ys = [p.intent_macro_f1 for p in sweep]
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=4,
            label="Routed (threshold sweep 0.00-1.00)")
    for res, color, marker, txt in [
        (small, ORANGE, "s", "Small only"),
        (large, AQUA, "D", "Large only"),
    ]:
        ax.scatter([res.cost_per_1k_usd], [res.intent_macro_f1], s=70, color=color,
                   marker=marker, edgecolor=SURFACE, linewidth=2, zorder=5, label=txt)
    for res, txt in [(default, f"default t={default.threshold:.2f}"),
                     (chosen, f"tuned t={chosen.threshold:.2f}")]:
        ax.scatter([res.cost_per_1k_usd], [res.intent_macro_f1], s=90, facecolor="none",
                   edgecolor=TEXT, linewidth=1.5, zorder=6)
        ax.annotate(f"{txt}\n{res.fallback_rate:.0%} fallback", (res.cost_per_1k_usd,
                    res.intent_macro_f1), textcoords="offset points", xytext=(10, -22),
                    fontsize=8, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 requests (USD, log scale)")
    ax.set_ylabel("Intent macro-F1 (test)")
    ax.set_title("Cost vs quality: small, large and routed", color=TEXT, fontsize=12,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def threshold_chart(sweep: list[StrategyResult], chosen: float, path: Path) -> None:
    """Two panels sharing x (threshold): quality and fallback rate - never a dual y-axis."""
    t = [p.threshold or 0.0 for p in sweep]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True,
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)
        ax.axvline(chosen, color=TEXT_2, linewidth=1, linestyle="--")
    ax1.plot(t, [p.intent_macro_f1 for p in sweep], color=BLUE, linewidth=2)
    ax1.set_ylabel("Intent macro-F1 (val)")
    ax1.set_title("Threshold tuning on the validation split", color=TEXT, fontsize=12,
                  loc="left")
    ax1.annotate(f"tuned t={chosen:.2f}", (chosen, ax1.get_ylim()[0]), xytext=(4, 6),
                 textcoords="offset points", fontsize=8, color=TEXT_2)
    ax2.plot(t, [p.fallback_rate for p in sweep], color=ORANGE, linewidth=2)
    ax2.set_ylabel("Fallback rate to large model")
    ax2.set_xlabel("Small-model confidence threshold")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def reliability_chart(conf: np.ndarray, correct: np.ndarray, path: Path, bins: int = 10) -> None:
    edges = np.linspace(0, 1, bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in itertools.pairwise(edges):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.sum() >= 5:
            centers.append(conf[mask].mean())
            accs.append(correct[mask].mean())
            counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.plot(centers, accs, color=BLUE, linewidth=2, marker="o", markersize=6,
            label="Small model (test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence (bin mean)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Small-model calibration", color=TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
