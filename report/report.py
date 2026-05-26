import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("eval/results/results.json")
CHARTS_DIR   = Path("eval/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_COLOR = "#10A37F"
OSS_COLOR    = "#FF6B35"

def load():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Run `python -m eval.run_eval` first.")
    return json.loads(RESULTS_PATH.read_text())


def avg(items, key):
    vals = [i[key] for i in items if i.get(key) is not None]
    return round(sum(vals) / len(vals), 1) if vals else 0


def chart_bars(results):
    """One grouped bar chart: avg overall_score per category per model."""
    categories = ["factual", "adversarial", "bias"]
    by = defaultdict(list)
    for r in results:
        by[(r["model"], r["category"])].append(r.get("overall_score", 0))

    openai_scores = [avg(results, "overall_score") if False else
                     round(sum(by[("openai", c)]) / max(len(by[("openai", c)]), 1), 1)
                     for c in categories]
    oss_scores = [round(sum(by[("oss_qwen", c)]) / max(len(by[("oss_qwen", c)]), 1), 1)
                  for c in categories]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - w/2, openai_scores, w, label="OpenAI GPT-4.1", color=OPENAI_COLOR, alpha=0.9)
    b2 = ax.bar(x + w/2, oss_scores,    w, label="OSS Qwen 2.5",   color=OSS_COLOR,    alpha=0.9)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{bar.get_height()}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["📋 Factual", "🛡️ Adversarial", "⚖️ Bias"], fontsize=12)
    ax.set_ylim(0, 11)
    ax.set_ylabel("Avg Score (0–10)", fontsize=11)
    ax.set_title("AI Assistant Evaluation — Overall Scores by Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(7, color="gray", linestyle="--", alpha=0.4, linewidth=1, label="Acceptable threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = CHARTS_DIR / "scores_by_category.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅ {out}")


def chart_latency(results):
    """Box plot of latency per model."""
    by_model = defaultdict(list)
    for r in results:
        if r.get("latency_s"):
            by_model[r["model"]].append(r["latency_s"])

    fig, ax = plt.subplots(figsize=(7, 5))
    models  = ["openai", "oss_qwen"]
    colors  = [OPENAI_COLOR, OSS_COLOR]
    labels  = ["OpenAI GPT-4.1", "OSS Qwen 2.5"]

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = by_model[model]
        if not vals:
            continue
        ax.boxplot(vals, positions=[i], widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="white", linewidth=2),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        ax.text(i, max(vals) + 0.1, f"avg: {np.mean(vals):.2f}s",
                ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title("⏱️ Response Latency Comparison", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = CHARTS_DIR / "latency.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅ {out}")


def main():
    print("\nGenerating")
    results = load()
    chart_bars(results)
    chart_latency(results)
    print(f"\nDone. Charts in {CHARTS_DIR}/")


if __name__ == "__main__":
    main()