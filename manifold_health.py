"""Manifold health checks before any GW alignment.

For each encoder (vision and audio, independently):
  - Singular-value scree + cumulative variance (elbow location).
  - 2D t-SNE colored by instrument class (visual sanity check).
  - Hubness: per-point k-occurrences distribution -> skewness (Radovanovic 2010).
  - Class isomorphism: between every pair of (vision, audio) encoders, compute
    Pearson and distance correlation between the 11x11 inter-class centroid
    distance matrices. Strong correlation = compatible class topology.
  - dCor between raw N x N cosine-distance matrices for every (vision, audio)
    pair. Tracks pre-decomposition global distance compatibility.

Writes:
  - manifold_health.json (all numbers)
  - plots/health_scree.png            (cumulative variance, all encoders)
  - plots/health_tsne_vision.png      (grid of t-SNEs, one per vision encoder)
  - plots/health_tsne_audio.png       (grid of t-SNEs, one per audio encoder)
  - plots/health_hubness.png          (skewness of k-occurrences, bar chart)
  - plots/health_isomorphism.png      (heatmap of inter-encoder class-centroid dCor)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from analyze import load_and_align

PLOTS = Path("plots")
OUT = Path("manifold_health.json")
SEED = 0
HUBNESS_K = 10


# ---------- core diagnostics ----------

def singular_values(X):
    """Singular values of centered X (sorted descending)."""
    Xc = X - X.mean(0, keepdims=True)
    return np.linalg.svd(Xc, compute_uv=False)


def cumvar(S):
    s2 = S.astype(np.float64) ** 2
    total = s2.sum()
    return s2.cumsum() / total if total > 1e-12 else s2


def elbow_index(S, threshold=0.9):
    """Index k such that cumulative variance >= threshold."""
    cv = cumvar(S)
    idx = int(np.searchsorted(cv, threshold)) + 1
    return min(idx, len(S))


def hubness_skewness(X, k=HUBNESS_K):
    """Skewness of the k-occurrence distribution (Radovanovic 2010).

    For each point i, count how many other points have i in their k-NN list.
    A heavy right tail (high skewness) means a few points are 'hubs' that
    are neighbors to everyone — bad for GW.
    """
    D = cosine_distances(X)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=min(k, D.shape[1] - 1), axis=1)[:, :k]
    counts = np.bincount(nn.ravel(), minlength=X.shape[0])
    return float(skew(counts)), counts


def class_centroid_distances(X, labels):
    """11x11 cosine distance matrix between class centroids."""
    classes = sorted(set(labels.tolist()))
    centroids = np.stack([X[labels == c].mean(0) for c in classes])
    return cosine_distances(centroids), classes


def distance_correlation(A, B):
    """Szekely's distance correlation between two square distance matrices."""
    assert A.shape == B.shape
    a_row = A.mean(axis=0, keepdims=True)
    a_col = A.mean(axis=1, keepdims=True)
    a_grand = A.mean()
    A_c = A - a_row - a_col + a_grand
    b_row = B.mean(axis=0, keepdims=True)
    b_col = B.mean(axis=1, keepdims=True)
    b_grand = B.mean()
    B_c = B - b_row - b_col + b_grand
    dCov2 = (A_c * B_c).mean()
    dVarA = (A_c * A_c).mean()
    dVarB = (B_c * B_c).mean()
    if dVarA <= 0 or dVarB <= 0:
        return 0.0
    return float(np.sqrt(max(dCov2, 0.0) / np.sqrt(dVarA * dVarB)))


def pearson_offdiag(A, B):
    iu = np.triu_indices_from(A, k=1)
    a = A[iu]; b = B[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------- plotting ----------

def plot_scree(per_encoder_S):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, S in per_encoder_S.items():
        ax.plot(np.arange(1, len(S) + 1), cumvar(S), label=name, alpha=0.8)
    ax.axhline(0.9, linestyle="--", color="grey", alpha=0.5, label="90% var")
    ax.set_xscale("log")
    ax.set_xlabel("component index")
    ax.set_ylabel("cumulative variance")
    ax.set_title("Spectral decay per encoder (centered SVD)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / "health_scree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_grid(per_encoder_X, labels, modality):
    names = list(per_encoder_X.keys())
    n = len(names)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    classes = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i / max(len(classes) - 1, 1)) for i, c in enumerate(classes)}
    for ax, name in zip(axes.ravel(), names):
        X = per_encoder_X[name]
        try:
            P = TSNE(n_components=2, random_state=SEED,
                     perplexity=min(30, max(2, len(X) // 3))).fit_transform(X)
        except Exception as e:
            ax.text(0.5, 0.5, f"t-SNE failed: {e}", ha="center", va="center")
            ax.set_title(name)
            continue
        for c in classes:
            mask = labels == c
            ax.scatter(P[mask, 0], P[mask, 1], s=18, alpha=0.8, color=colors[c], label=c)
        ax.set_title(name); ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[c], label=c)
               for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(11, len(classes)), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"t-SNE per {modality} encoder", fontsize=12)
    plt.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / f"health_tsne_{modality}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_hubness(skews):
    """skews: dict name -> skewness scalar."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(skews.keys())
    vals = [skews[n] for n in names]
    bars = ax.bar(range(len(names)), vals, alpha=0.85)
    for bar, v in zip(bars, vals):
        bar.set_color("crimson" if v > 1.5 else "steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(1.5, linestyle="--", color="grey", alpha=0.5, label="warn: skew > 1.5")
    ax.set_ylabel(f"skewness of {HUBNESS_K}-occurrences")
    ax.set_title("Hubness per encoder (higher = a few points dominate k-NN)")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend()
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / "health_hubness.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_isomorphism_heatmap(matrix, vnames, anames, title, fname):
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(anames), 1 + 0.5 * len(vnames)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(anames))); ax.set_xticklabels(anames, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(vnames))); ax.set_yticklabels(vnames, fontsize=8)
    for i in range(len(vnames)):
        for j in range(len(anames)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i,j] < 0.5 else "black", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- driver ----------

def main():
    aligned, _ = load_and_align(["vision", "audio"])
    vision, audio = aligned["vision"], aligned["audio"]
    if not vision or not audio:
        raise SystemExit("no embeddings found; run encode.py first")

    labels = next(iter(vision.values()))["labels"]
    vnames = list(vision.keys())
    anames = list(audio.keys())

    # 1) scree
    print("== scree ==")
    per_S = {}
    elbows = {}
    for name, d in {**{f"vision/{n}": v for n, v in vision.items()},
                    **{f"audio/{n}": v for n, v in audio.items()}}.items():
        S = singular_values(d["X"])
        per_S[name] = S
        elbows[name] = elbow_index(S, threshold=0.9)
        print(f"  {name:30s}  d={d['X'].shape[1]:4d}  elbow(90%)={elbows[name]:3d}")
    plot_scree(per_S)

    # 2) t-SNE side-by-side
    print("== t-SNE ==")
    plot_tsne_grid({n: v["X"] for n, v in vision.items()}, labels, "vision")
    plot_tsne_grid({n: v["X"] for n, v in audio.items()}, labels, "audio")

    # 3) hubness
    print("== hubness ==")
    skews = {}
    hub_top = {}
    for name, d in {**{f"vision/{n}": v for n, v in vision.items()},
                    **{f"audio/{n}": v for n, v in audio.items()}}.items():
        sk, counts = hubness_skewness(d["X"])
        skews[name] = sk
        hub_top[name] = int(counts.max())
        print(f"  {name:30s}  skew={sk:+.3f}  max_kocc={counts.max()}  (warn>1.5)")
    plot_hubness(skews)

    # 4) class isomorphism: class-centroid 11x11 distance matrices, all pairs
    print("== class-centroid isomorphism ==")
    v_centroids = {n: class_centroid_distances(v["X"], labels)[0] for n, v in vision.items()}
    a_centroids = {n: class_centroid_distances(v["X"], labels)[0] for n, v in audio.items()}
    iso_dcor = np.zeros((len(vnames), len(anames)))
    iso_pear = np.zeros_like(iso_dcor)
    for i, vn in enumerate(vnames):
        for j, an in enumerate(anames):
            iso_dcor[i, j] = distance_correlation(v_centroids[vn], a_centroids[an])
            iso_pear[i, j] = pearson_offdiag(v_centroids[vn], a_centroids[an])
        print(f"  {vn:15s}  best: " +
              ", ".join(f"{an}={iso_dcor[i,j]:.2f}" for j, an in enumerate(anames)))
    plot_isomorphism_heatmap(iso_dcor, vnames, anames,
                             "Class-centroid dCor (vision × audio)",
                             "health_isomorphism_centroid_dcor.png")
    plot_isomorphism_heatmap(iso_pear, vnames, anames,
                             "Class-centroid Pearson (vision × audio)",
                             "health_isomorphism_centroid_pearson.png")

    # 5) raw N x N dCor pre-decomposition (every pair)
    print("== raw N x N dCor ==")
    raw_dcor = np.zeros((len(vnames), len(anames)))
    v_full = {n: cosine_distances(v["X"]) for n, v in vision.items()}
    a_full = {n: cosine_distances(v["X"]) for n, v in audio.items()}
    for i, vn in enumerate(vnames):
        for j, an in enumerate(anames):
            raw_dcor[i, j] = distance_correlation(v_full[vn], a_full[an])
    plot_isomorphism_heatmap(raw_dcor, vnames, anames,
                             "Raw N×N cosine-distance dCor (vision × audio)",
                             "health_raw_dcor.png")

    # write json
    payload = {
        "elbows_90pct": elbows,
        "hubness_skewness": skews,
        "hubness_max_kocc": hub_top,
        "centroid_dcor": {vn: {an: float(iso_dcor[i, j]) for j, an in enumerate(anames)}
                          for i, vn in enumerate(vnames)},
        "centroid_pearson": {vn: {an: float(iso_pear[i, j]) for j, an in enumerate(anames)}
                             for i, vn in enumerate(vnames)},
        "raw_dcor": {vn: {an: float(raw_dcor[i, j]) for j, an in enumerate(anames)}
                     for i, vn in enumerate(vnames)},
        "vnames": vnames, "anames": anames,
        "hubness_k": HUBNESS_K,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
