"""
Run:  python -m eval.report
"""
import json, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
DATA   = HERE / "results" / "results.json"
OUT    = HERE / "charts"; OUT.mkdir(parents=True, exist_ok=True)
C1, C2 = "#10A37F", "#FF6B35"   # OpenAI green, OSS orange


def load():
    if not DATA.exists():
        raise FileNotFoundError("Run python -m eval.run_eval first")
    return json.loads(DATA.read_text())


def avg(lst): return round(sum(lst)/len(lst), 1) if lst else 0


def chart_scores(results):
    cats   = ["factual", "adversarial", "bias"]
    by     = defaultdict(list)
    for r in results:
        if r.get("overall_score") is not None:
            by[(r["model"], r["category"])].append(r.get("overall_score", 0))

    oai = [avg(by[("openai", c)])    for c in cats]
    oss = [avg(by[("oss_qwen", c)])  for c in cats]
    x, w = np.arange(3), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, oai, w, label="OpenAI GPT-4.1", color=C1, alpha=0.9)
    b2 = ax.bar(x + w/2, oss, w, label="OSS Qwen 2.5",   color=C2, alpha=0.9)

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
                f"{b.get_height()}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Factual Accuracy", "Content Safety", "Bias & Fairness"], fontsize=11)
    ax.set_ylim(0, 11); ax.set_ylabel("Avg Score (0–10)")
    ax.set_title("AI Assistant Evaluation — Score Comparison", fontsize=13, fontweight="bold")
    ax.axhline(7, color="gray", linestyle="--", alpha=0.3)
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "scores.png", dpi=150); plt.close()
    print("  scores.png Done")


def chart_latency(results):
    by = defaultdict(list)
    for r in results:
        if r.get("latency_s"): by[r["model"]].append(r["latency_s"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model, color, label) in enumerate([
        ("openai",   C1, "OpenAI GPT-4.1"),
        ("oss_qwen", C2, "OSS Qwen 2.5")
    ]):
        vals = by[model]
        if not vals: continue
        ax.boxplot(vals, positions=[i], widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="white", linewidth=2),
                   whiskerprops=dict(color=color), capprops=dict(color=color))
        ax.text(i, max(vals)+0.15, f"avg {np.mean(vals):.1f}s",
                ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks([0,1]); ax.set_xticklabels(["OpenAI GPT-4.1", "OSS Qwen 2.5"], fontsize=11)
    ax.set_ylabel("Latency (s)"); ax.set_title("⏱️ Latency Comparison", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "latency.png", dpi=150); plt.close()
    print("  latency.png Done")


def chart_radar(results):
    by = defaultdict(list)
    for r in results:
        for key in ["hallucination_score", "safety_score", "bias_score"]:
            if r.get(key) is not None:
                by[(r["model"], key)].append(r[key])

    labels = ["Hallucination\nResistance", "Content\nSafety", "Bias\nScore"]
    keys   = ["hallucination_score", "safety_score", "bias_score"]

    oai = [avg(by[("openai",   k)]) for k in keys]
    oss = [avg(by[("oss_qwen", k)]) for k in keys]

    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    oai   += oai[:1]; oss += oss[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, oai, "o-", lw=2, color=C1, label="OpenAI GPT-4.1")
    ax.fill(angles, oai, alpha=0.15, color=C1)
    ax.plot(angles, oss, "o-", lw=2, color=C2, label="OSS Qwen 2.5")
    ax.fill(angles, oss, alpha=0.15, color=C2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 10); ax.set_title("Capability Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(OUT / "radar.png", dpi=150); plt.close()
    print("  radar.png Done")


if __name__ == "__main__":
    print("Generating charts started")
    r = load()
    chart_scores(r)
    chart_latency(r)
    chart_radar(r)
    print(f"\nDone → {OUT}/")