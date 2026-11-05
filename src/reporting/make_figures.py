"""
Charts for the README, drawn from the report files a run already produced.

    python -m src.reporting.make_figures

Reads  reports/decile_table_holdout.csv, reports/phase3_model_comparison.csv,
       reports/shap_importance.csv
Writes docs/images/*.png (committed to git; reports/ is not).

Only AGGREGATE figures are drawn: rates per risk band, model scores, mean feature
influence. Nothing here shows an individual applicant, so the images can be published
without republishing any of the dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

NAVY, RED, GREEN, ORANGE, GREY = "#243b73", "#b3261e", "#1f7a4d", "#d98324", "#6e6b62"
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "axes.titleweight": "bold", "figure.facecolor": "white"})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def overall_rate(decile_csv: Path) -> float:
    d = pd.read_csv(decile_csv)
    return float((d["observed"] * d["applicants"]).sum() / d["applicants"].sum())


def risk_bands(decile_csv: Path, out: Path) -> Path:
    d = pd.read_csv(decile_csv)
    avg = overall_rate(decile_csv)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(d["risk_decile"], d["observed"], color=NAVY, label="Observed default rate", zorder=2)
    ax.plot(d["risk_decile"], d["predicted"], "o", color=ORANGE, markersize=7, label="Model's predicted rate", zorder=3)
    ax.axhline(avg, color=GREY, linestyle="--", linewidth=1)
    ax.text(0.55, avg + 0.006, f"average {avg:.1%}", color=GREY, fontsize=9, va="bottom")
    for x, o, pr in zip(d["risk_decile"], d["observed"], d["predicted"]):
        ax.text(x, max(o, pr) + 0.008, f"{o:.1%}", ha="center", fontsize=8.5, color="#222222")
    ax.set_xticks(d["risk_decile"])
    ax.set_xlabel("Risk band (1 = safest tenth of applicants, 10 = riskiest)")
    ax.set_ylabel("Default rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(d["observed"].max(), d["predicted"].max()) * 1.18)
    ax.set_title("Holdout set: default rate by risk band", loc="left")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out)


def model_comparison(csv: Path, out: Path, baseline_pr: float | None = None) -> Path:
    d = pd.read_csv(csv).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    labels = [f"{m}\n{i.replace('pos_weight', 'class-weighted').replace('none', 'unweighted')}"
              for m, i in zip(d["model"], d["imbalance"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title, lo in ((axes[0], "pr_auc", "PR-AUC (selection metric)", 0.0), (axes[1], "roc_auc", "ROC-AUC", 0.5)):
        ax.bar(range(len(d)), d[col], color=NAVY, zorder=2)
        for i, v in enumerate(d[col]):
            ax.text(i, v + (d[col].max() - lo) * 0.015, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(lo, d[col].max() + (d[col].max() - lo) * 0.18)
        ax.set_title(title, loc="left")
    if baseline_pr is not None:
        axes[0].axhline(baseline_pr, color=ORANGE, linestyle="--", linewidth=1.6, zorder=3,
                        label=f"random ranker ({baseline_pr:.3f})")
        axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right", fontsize=8.5)
    axes[1].set_xlabel("axis starts at 0.5, the score of a random ranker", fontsize=8.5, color=GREY)
    fig.suptitle("Four candidate models on the validation set: all within "
                 f"{d['pr_auc'].max() - d['pr_auc'].min():.3f} PR-AUC", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def top_features(csv: Path, out: Path, n: int = 12) -> Path:
    d = pd.read_csv(csv).sort_values("mean_abs_shap", ascending=False).head(n).iloc[::-1]
    names = d["label"] if "label" in d else d["feature"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(names, d["mean_abs_shap"], color=NAVY, zorder=2)
    ax.set_xlabel("Average influence on the score (mean |SHAP|, log-odds)")
    ax.set_title("Most influential features", loc="left")
    return _save(fig, out)


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none", alpha=0.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="white", fontweight="bold", linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#444444",
                                 linewidth=1.2, connectionstyle=f"arc3,rad={rad}"))


def architecture(out: Path) -> Path:
    DATA, FEAT, MODEL, SERVE, SIDE = "#243b73", "#1f7a4d", "#b3651d", "#6a3d9a", "#6f6e66"
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    cols, w, h = [0.1, 2.95, 5.8, 8.65], 2.4, 0.85
    ys = {0: 5.45, 1: 4.15, 2: 2.85, 3: 1.55, 4: 0.25}

    def box(c, r, text, color):
        _box(ax, cols[c], ys[r], w, h, text, color)

    def mid(c, r):
        return cols[c] + w / 2, ys[r] + h / 2

    box(0, 1, "Kaggle CSVs\n3 tables", SIDE)
    box(1, 1, "Bronze (Delta)\nraw rows + metadata", DATA)
    box(2, 1, "Validation\nGreat Expectations\n+ integrity checks", DATA)
    box(3, 1, "Silver (Delta)\nclean, validated rows", DATA)
    box(2, 0, "Quarantine\nfailed rows + reasons", SIDE)
    for c in range(3):
        _arrow(ax, cols[c] + w, ys[1] + h / 2, cols[c + 1], ys[1] + h / 2)
    _arrow(ax, mid(2, 1)[0], ys[1] + h, mid(2, 0)[0], ys[0])

    box(3, 2, "PySpark features\n38 model inputs", FEAT)
    box(2, 2, "Gold (Delta)\n+ Parquet export", FEAT)
    box(1, 2, "Feast feature store\noffline | online", FEAT)
    _arrow(ax, mid(3, 1)[0], ys[1], mid(3, 2)[0], ys[2] + h)
    _arrow(ax, cols[3], ys[2] + h / 2, cols[2] + w, ys[2] + h / 2)
    _arrow(ax, cols[2], ys[2] + h / 2, cols[1] + w, ys[2] + h / 2)

    box(0, 3, "Training\nLightGBM, holdout\nscored once", MODEL)
    box(1, 3, "MLflow registry\nchampion + threshold", MODEL)
    box(2, 3, "FastAPI /score\n+ scorecard page", SERVE)
    box(3, 3, "Batch scoring\nDelta + CSV export", SERVE)
    _arrow(ax, cols[1] + 0.5, ys[2], cols[0] + w - 0.5, ys[3] + h, rad=0.15)
    _arrow(ax, cols[0] + w, ys[3] + h / 2, cols[1], ys[3] + h / 2)
    _arrow(ax, cols[1] + w, ys[3] + h / 2, cols[2], ys[3] + h / 2)
    _arrow(ax, cols[1] + w - 0.4, ys[2], cols[2] + 0.5, ys[3] + h, rad=-0.15)
    _arrow(ax, cols[2] + w, ys[3] + h / 2, cols[3], ys[3] + h / 2)

    box(2, 4, "Parity check\noffline = online", SIDE)
    box(3, 4, "Drift monitoring\nPSI per feature + score", SIDE)
    _arrow(ax, mid(2, 3)[0], ys[3], mid(2, 4)[0], ys[4] + h)
    _arrow(ax, mid(3, 3)[0], ys[3], mid(3, 4)[0], ys[4] + h)

    for i, (name, color) in enumerate((("Data", DATA), ("Features", FEAT), ("Model", MODEL), ("Serving", SERVE))):
        ax.text(0.15 + (i % 2) * 1.5, 0.75 - (i // 2) * 0.35, "\u25A0 " + name, color=color, fontweight="bold", fontsize=10.5)
    return _save(fig, out)


def main(reports: Path | None = None, images: Path | None = None) -> int:
    from src.common import config

    reports = reports or config.REPORTS_DIR
    images = images or (config.PROJECT_ROOT / "docs" / "images")
    made, skipped = [], []
    made.append(architecture(images / "architecture.png"))
    deciles = reports / "decile_table_holdout.csv"
    jobs = [
        ("risk_bands.png", deciles, lambda p, o: risk_bands(p, o)),
        ("model_comparison.png", reports / "phase3_model_comparison.csv",
         lambda p, o: model_comparison(p, o, overall_rate(deciles) if deciles.exists() else None)),
        ("top_features.png", reports / "shap_importance.csv", lambda p, o: top_features(p, o)),
    ]
    for name, src, fn in jobs:
        if src.exists():
            made.append(fn(src, images / name))
        else:
            skipped.append(f"{name} (missing {src.name})")
    for m in made:
        print("wrote", m)
    for s in skipped:
        print("skipped", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
