"""Read comparison_matrix.json, print ranked summary, save interpretation plots.

Usage: python report.py
Outputs:
  stdout — ranked summary table, headline numbers, automatic interpretation
  plots/report_bars_<metric>.png       — methods compared at the headline k
  plots/report_ksweep_<metric>.png     — metric vs k for each method (avg across pairs)
  plots/report_heatmap_<method>.png    — per-pair heatmap of recall@10 for selected methods
  plots/report_paired_vs_unsup.png     — scatter of paired vs unsupervised on the same pair
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("comparison_matrix.json")
PLOTS = Path("plots")
HEADLINE_K = 5
PRIMARY_METRIC = "recall_at_11"   # 11 = n_classes; class-level chance ≈ 0.66


def chance_recall_at_k(per_class, n_test, k):
    """Hypergeometric chance for recall@k: any of k random draws is same-class."""
    same = per_class
    other = n_test - same
    if k > other:
        return 1.0
    p_none = 1.0
    for i in range(k):
        p_none *= (other - i) / (n_test - i)
    return 1.0 - p_none


def chance_baselines(n_test=242, n_class=11):
    """Defaults assume the full-data unsupervised setup (no train/test split)."""
    per = n_test // n_class
    # MRR chance for uniform-random ranking ≈ (1 + 1/2 + ... + 1/N) / N
    mrr_chance = float(np.mean(1.0 / np.arange(1, n_test + 1)))
    return {
        "class_purity":     per / n_test,
        "recall_at_5":      chance_recall_at_k(per, n_test, 5),
        "recall_at_11":     chance_recall_at_k(per, n_test, 11),
        "recall_at_22":     chance_recall_at_k(per, n_test, 22),
        "instance_r_at_1":  1 / n_test,
        "instance_r_at_10": 10 / n_test,
        "hungarian_purity": float("nan"),  # depends on N, classes, not closed-form
        "foscttm":          0.5,
        "gw_cost":          float("nan"),
        "dcor_postdecomp":  float("nan"),
        "mrr":              mrr_chance,
        "nmi":              0.0,            # NMI of random labeling ≈ 0
        "cka":              float("nan"),   # depends on data; no closed-form
    }


def load_rows(path=RESULTS):
    rows = json.loads(path.read_text())
    # Normalize: k may be None or int. method, vision, audio, supervision present.
    for r in rows:
        r.setdefault("supervision", "-")
    return rows


# ------------------- summarization -------------------

def group_by_method_at_k(rows, k, metric):
    """For each (method, supervision), collect metric values across (vision, audio) pairs.

    For no-k methods (procrustes, plain_gw), they only appear once per pair (k=None);
    include them whenever k is specified to surface them in the summary.
    """
    out = defaultdict(list)
    for r in rows:
        rk = r.get("k")
        if rk == k or rk is None:
            v = r.get(metric)
            if isinstance(v, (int, float)) and not np.isnan(v):
                out[(r["method"], r["supervision"])].append(v)
    return out


def print_summary(rows, k=HEADLINE_K, metric=PRIMARY_METRIC):
    chance = chance_baselines().get(metric, float("nan"))
    grouped = group_by_method_at_k(rows, k, metric)
    rankings = sorted(
        ((m, s, np.mean(v), np.std(v), np.min(v), np.max(v), len(v))
         for (m, s), v in grouped.items()),
        key=lambda x: -x[2],
    )
    print(f"\n=== ranked by mean {metric} (k={k} for k-methods; chance={chance:.3f}) ===")
    print(f"{'rank':<5} {'method':<22} {'sup':<8} {'mean':<8} {'std':<8} {'min':<8} {'max':<8} {'n':<4}")
    print("-" * 80)
    for i, (m, s, mean, std, mn, mx, n) in enumerate(rankings, 1):
        flag = "  ↑" if mean > chance else ""
        print(f"{i:<5} {m:<22} {s:<8} {mean:.3f}    {std:.3f}    {mn:.3f}    {mx:.3f}    {n:<4}{flag}")
    return rankings


def print_interpretation(rows, k=HEADLINE_K, metric=PRIMARY_METRIC):
    chance = chance_baselines().get(metric, float("nan"))
    grouped = group_by_method_at_k(rows, k, metric)
    by_sup = defaultdict(list)
    for (m, s), vals in grouped.items():
        by_sup[s].append((m, float(np.mean(vals))))

    print(f"\n=== automatic interpretation ({metric}, k={k}, chance={chance:.3f}) ===")
    for tier in ("paired", "none"):
        if tier not in by_sup:
            continue
        best_m, best_mean = max(by_sup[tier], key=lambda x: x[1])
        cnt_above = sum(1 for _, mean in by_sup[tier] if mean > chance)
        print(f"  best {tier:6s} method: {best_m:20s}  mean={best_mean:.3f}  "
              f"({cnt_above}/{len(by_sup[tier])} methods beat chance)")

    if "paired" in by_sup and "none" in by_sup:
        best_p = max(m for m in by_sup["paired"]   if not np.isnan(m[1]) for m in [m])[1] if False else max(by_sup["paired"], key=lambda x: x[1])[1]
        best_u = max(by_sup["none"],   key=lambda x: x[1])[1]
        gap = best_p - best_u
        print(f"  best-paired − best-unsupervised gap: {gap:+.3f}")
        if best_u > chance + 0.05:
            print("  → unsupervised methods exceed chance; weak Platonic support")
        else:
            print("  → unsupervised methods at/below chance; strong claim NOT supported")
        if best_p > chance + 0.10:
            print("  → paired methods clearly work for retrieval")


# ------------------- plots -------------------

def plot_method_bars(rows, k=HEADLINE_K, metric=PRIMARY_METRIC):
    grouped = group_by_method_at_k(rows, k, metric)
    if not grouped:
        print(f"  skip bars for {metric}: no rows with this metric at k={k}")
        return
    chance = chance_baselines().get(metric, float("nan"))
    items = sorted(grouped.items(), key=lambda x: -np.mean(x[1]))
    names = [f"{m}\n[{s}]" for (m, s), _ in items]
    means = [float(np.mean(v)) for _, v in items]
    stds = [float(np.std(v)) for _, v in items]
    colors = ["#1f77b4" if s == "none" else "#ff7f0e" for (_, s), _ in items]
    n_pairs = len({(r["vision"], r["audio"]) for r in rows})

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(items)), 5))
    ax.bar(range(len(items)), means, yerr=stds, color=colors, alpha=0.85, capsize=4)
    if not np.isnan(chance):
        ax.axhline(chance, linestyle="--", color="grey",
                   label=f"chance = {chance:.3f}")
    ax.set_xticks(range(len(items))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per method (k={k}; mean ± std across {n_pairs} pairs)")
    ymax = max(1.0, max(means) + max(stds) + 0.1) if means else 1.0
    ax.set_ylim(0, ymax); ax.grid(True, axis="y", alpha=0.3)
    # supervision legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#1f77b4", label="unsupervised"),
               Patch(facecolor="#ff7f0e", label="paired")]
    if not np.isnan(chance):
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="grey", linestyle="--", label=f"chance"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    out = PLOTS / f"report_bars_{metric}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")


def plot_k_sweep(rows, metric=PRIMARY_METRIC):
    chance = chance_baselines().get(metric, float("nan"))
    n_pairs = len({(r["vision"], r["audio"]) for r in rows})
    sup_of = {}
    by_method_k = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("k") is None:
            continue
        v = r.get(metric)
        if isinstance(v, (int, float)) and not np.isnan(v):
            by_method_k[r["method"]][r["k"]].append(v)
            sup_of[r["method"]] = r["supervision"]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_paired = plt.get_cmap("Oranges")
    cmap_unsup  = plt.get_cmap("Blues")
    paired_methods = sorted([m for m, s in sup_of.items() if s == "paired"])
    unsup_methods  = sorted([m for m, s in sup_of.items() if s == "none"])
    for i, m in enumerate(paired_methods):
        ks = sorted(by_method_k[m]); ys = [float(np.mean(by_method_k[m][k])) for k in ks]
        ax.plot(ks, ys, marker="o", color=cmap_paired(0.4 + 0.5 * i / max(1, len(paired_methods) - 1)), label=f"{m} [paired]")
    for i, m in enumerate(unsup_methods):
        ks = sorted(by_method_k[m]); ys = [float(np.mean(by_method_k[m][k])) for k in ks]
        ax.plot(ks, ys, marker="s", linestyle="--", color=cmap_unsup(0.4 + 0.5 * i / max(1, len(unsup_methods) - 1)), label=f"{m} [unsup]")

    if not np.isnan(chance):
        ax.axhline(chance, linestyle=":", color="black", alpha=0.5, label=f"chance = {chance:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("k")
    ax.set_ylabel(f"mean {metric} (averaged over {n_pairs} pairs)")
    ax.set_title(f"{metric} vs k by method")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / f"report_ksweep_{metric}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")


def plot_pair_heatmap(rows, method, k, metric=PRIMARY_METRIC):
    chance = chance_baselines().get(metric, float("nan"))
    visions = sorted({r["vision"] for r in rows})
    audios  = sorted({r["audio"]  for r in rows})
    M = np.full((len(visions), len(audios)), np.nan)
    for r in rows:
        rk = r.get("k")
        if r["method"] == method and (rk == k or (rk is None and k is None)):
            i = visions.index(r["vision"]); j = audios.index(r["audio"])
            v = r.get(metric)
            if isinstance(v, (int, float)):
                M[i, j] = v
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(audios))); ax.set_xticklabels(audios, rotation=30, ha="right")
    ax.set_yticks(range(len(visions))); ax.set_yticklabels(visions)
    suffix = f"k={k}" if k is not None else "no-k"
    ax.set_title(f"{method} ({suffix}) — {metric} per pair (chance={chance:.3f})")
    for i in range(len(visions)):
        for j in range(len(audios)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.4 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    out = PLOTS / f"report_heatmap_{method}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")


def plot_paired_vs_unsupervised(rows, k=HEADLINE_K, metric=PRIMARY_METRIC):
    """Per-pair scatter: best unsupervised method r@10 vs best paired method r@10."""
    by_pair = defaultdict(lambda: {"paired": [], "none": []})
    for r in rows:
        if r.get("k") == k or r.get("k") is None:
            v = r.get(metric)
            if isinstance(v, (int, float)) and not np.isnan(v):
                by_pair[(r["vision"], r["audio"])][r["supervision"]].append(v)
    xs, ys, lbls = [], [], []
    for pair, d in by_pair.items():
        if d["paired"] and d["none"]:
            xs.append(max(d["none"])); ys.append(max(d["paired"]))
            lbls.append(f"{pair[0]}×{pair[1]}")
    chance = chance_baselines().get(metric, float("nan"))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xs, ys, s=70, alpha=0.7)
    for x, y, l in zip(xs, ys, lbls):
        ax.annotate(l, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    lo = min(min(xs), min(ys)) - 0.05
    hi = max(max(xs), max(ys)) + 0.05
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", label="y = x")
    ax.axvline(chance, linestyle=":", color="black", alpha=0.4, label=f"chance ({chance:.3f})")
    ax.axhline(chance, linestyle=":", color="black", alpha=0.4)
    ax.set_xlabel(f"best UNSUPERVISED method {metric}")
    ax.set_ylabel(f"best PAIRED method {metric}")
    ax.set_title(f"Paired vs Unsupervised at k={k} (one point per (vision, audio) pair)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    out = PLOTS / f"report_paired_vs_unsup_{metric}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")


# ------------------- main -------------------

def main():
    rows = load_rows()
    print(f"loaded {len(rows)} rows from {RESULTS}")
    print(f"unique pairs:   {len({(r['vision'], r['audio']) for r in rows})}")
    print(f"unique methods: {len({r['method'] for r in rows})}")
    PLOTS.mkdir(exist_ok=True)

    # Headline text summary
    print_summary(rows, k=HEADLINE_K, metric=PRIMARY_METRIC)
    print_summary(rows, k=11, metric=PRIMARY_METRIC)
    print_interpretation(rows, k=HEADLINE_K, metric=PRIMARY_METRIC)

    # Plots
    for metric in ("recall_at_11", "recall_at_5", "recall_at_22",
                   "hungarian_purity", "instance_r_at_1",
                   "foscttm", "dcor_postdecomp",
                   "mrr", "nmi", "cka"):
        plot_method_bars(rows, k=HEADLINE_K, metric=metric)
        plot_k_sweep(rows, metric=metric)

    # Per-method pair heatmaps for the new method set
    rp_k = next((r["k"] for r in rows if r["method"] == "random_proj"), None)
    heatmap_specs = [
        ("vanilla",         None),
        ("random_proj",     rp_k),
        ("svd_truncate",    HEADLINE_K),
        ("spectral_whiten", HEADLINE_K),
        ("kpca_rbf",        HEADLINE_K),
        ("wprocrustes",     HEADLINE_K),
        ("spectral_gw",     HEADLINE_K),
        ("spectral_gw_mr",  HEADLINE_K),
    ]
    for method, k in heatmap_specs:
        if any(r["method"] == method for r in rows):
            plot_pair_heatmap(rows, method, k)


if __name__ == "__main__":
    main()
