"""Cross-modal retrieval via CCA-projected representations, with held-out evaluation.

For each (vision_model, audio_model) pair:
  - Stratified train/test split (80/20 by instrument).
  - Fit any cross-modal projection on TRAIN; evaluate retrieval on TEST.
  - Compare four methods on the same test split:
      1. plain_gw      — entropic GW on raw test embeddings (no projection)
      2. procrustes    — least-squares linear map audio->vision, fit on train, cosine on test
      3. cca_cos       — CCA-projected to top-k shared subspace, cosine on test
      4. cca_gw        — CCA-projected to top-k shared subspace, entropic GW on test
  - Reports class_purity, hungarian_purity, recall@{5,10}, instance@{1,10}.

Also produces macro-cluster plots and a singular-value-decay plot.
"""
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from analyze import (
    class_purity,
    cost_matrix,
    gw_entropic,
    hungarian_purity,
    instance_recall_at_k,
    load_and_align,
    recall_at_k,
)

OUT = Path("cca_results.json")
PLOTS = Path("plots")
K_VALUES = [5, 11, 22]
K_HEADLINE = 5
TEST_FRAC = 0.2
SEED = 0
EPS_ENTROPIC = 0.05  # matches analyze.gw_entropic's epsilon


# ---------- split, projections, scores ----------

def stratified_split(labels, test_frac, seed):
    """Per-class stratified split. Returns (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.array(sorted(train)), np.array(sorted(test))


def cross_cov_svd(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return U, S, Vt.T


def variance_captured(S, k):
    s2 = S.astype(np.float64) ** 2
    return float(s2[:k].sum() / s2.sum()) if s2.sum() > 1e-12 else 0.0


def cosine_scores(A, B, eps=1e-12):
    """Pairwise cosine-similarity matrix: A_normalized @ B_normalized.T."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A @ B.T


def fit_procrustes(X_train, Y_train):
    """Least-squares linear map W: audio space -> vision space.

    W minimizes ||X_train @ W - Y_train||_F. No orthogonality constraint
    because the modality dims differ. Standard 'stitching' baseline.
    """
    # np.linalg.lstsq handles d_x != d_y. Returns (W, residuals, rank, sv).
    W, *_ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    return W


def _inv_sqrt(M, eps=1e-10):
    """Inverse square root of a symmetric positive semi-definite matrix."""
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ (V.T * (1.0 / np.sqrt(w))[:, None])


def fit_cca(X_train, Y_train, ridge_frac=1e-2):
    """Ridge-regularized CCA.

    Whitens each modality by its own within-modality covariance, then SVDs the
    whitened cross-covariance. Components have unit within-modality variance and
    singular values are exactly the cross-modal canonical correlations in [0,1].

    With N < d (under-determined), ridge regularization is essential. Ridge
    strength is set adaptively as a fraction of each covariance's mean eigenvalue,
    so it's scale-invariant across modalities.

    Returns (A, B, rho) — CCA bases for X and Y (apply to centered data), and
    the canonical correlations.
    """
    n = len(X_train)
    Xc = X_train - X_train.mean(0, keepdims=True)
    Yc = Y_train - Y_train.mean(0, keepdims=True)
    Cxx = (Xc.T @ Xc) / n
    Cyy = (Yc.T @ Yc) / n
    Cxy = (Xc.T @ Yc) / n
    Cxx = Cxx + (ridge_frac * np.trace(Cxx) / Cxx.shape[0]) * np.eye(Cxx.shape[0])
    Cyy = Cyy + (ridge_frac * np.trace(Cyy) / Cyy.shape[0]) * np.eye(Cyy.shape[0])
    Wx = _inv_sqrt(Cxx)
    Wy = _inv_sqrt(Cyy)
    U, rho, Vt = np.linalg.svd(Wx @ Cxy @ Wy, full_matrices=False)
    A = Wx @ U          # audio CCA basis
    B = Wy @ Vt.T       # vision CCA basis
    return A, B, rho


# ---------- truly unsupervised projections (no paired correspondences) ----------

def fit_pca(X_train, k):
    """Independent PCA on X. Uses only within-modality variance."""
    mu = X_train.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
    return Vt.T[:, :k], mu.squeeze()


def fit_wprocrustes(X_a, X_b, n_iters=30, sinkhorn_reg=0.05, seed=SEED):
    """Wasserstein Procrustes (Grave et al. 2018).

    Iteratively alternate Sinkhorn OT and orthogonal Procrustes to find a
    rotation R aligning two point clouds *without* paired correspondences.
    X_a and X_b must already share dimensionality (PCA-reduce beforehand).
    """
    rng = np.random.default_rng(seed)
    n_a, d = X_a.shape
    n_b = len(X_b)
    R = np.linalg.qr(rng.standard_normal((d, d)))[0]
    p = np.full(n_a, 1.0 / n_a)
    q = np.full(n_b, 1.0 / n_b)
    for _ in range(n_iters):
        XR = X_a @ R
        # squared euclidean cost matrix, normalized for Sinkhorn stability
        sq_a = (XR ** 2).sum(1)
        sq_b = (X_b ** 2).sum(1)
        C = sq_a[:, None] + sq_b[None, :] - 2 * XR @ X_b.T
        C = np.maximum(C, 0.0)
        C = C / max(C.max(), 1e-12)
        P = ot.sinkhorn(p, q, C, reg=sinkhorn_reg, numItermax=200, stopThr=1e-6)
        # Procrustes step: best orthogonal R given coupling P
        M = X_a.T @ P @ X_b
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
    return R


def spectral_embed(Z, k, eps=1e-12):
    """Normalized-Laplacian spectral embedding via RBF kernel on cosine distance.

    Bandwidth = median pairwise distance. Returns N x k embedding (rows of
    top-k eigenvectors of the symmetrically-normalized kernel matrix).
    """
    D = cosine_distances(Z)
    sigma = max(np.median(D[D > 0]), eps)
    K = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(K.sum(axis=1), eps))
    K_norm = K * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    _, V = np.linalg.eigh(K_norm)
    # eigh returns ascending; take the top (largest eigenvalue) k columns
    return V[:, -k:]


# ---------- four retrieval methods, each returns a (n_test x n_test) score matrix ----------

def method_plain_gw(X_train, Y_train, X_test, Y_test, **kw):
    pi, _ = gw_entropic(cost_matrix(X_test), cost_matrix(Y_test))
    return pi


def method_procrustes(X_train, Y_train, X_test, Y_test, **kw):
    W = fit_procrustes(X_train, Y_train)
    X_in_Y = X_test @ W
    return cosine_scores(X_in_Y, Y_test)


def method_cca_cos(X_train, Y_train, X_test, Y_test, k, **kw):
    U, S, V = cross_cov_svd(X_train, Y_train)
    mu_x, mu_y = X_train.mean(0), Y_train.mean(0)
    Xp = (X_test - mu_x) @ U[:, :k]
    Yp = (Y_test - mu_y) @ V[:, :k]
    return cosine_scores(Xp, Yp)


def method_cca_gw(X_train, Y_train, X_test, Y_test, k, **kw):
    U, S, V = cross_cov_svd(X_train, Y_train)
    mu_x, mu_y = X_train.mean(0), Y_train.mean(0)
    Xp = (X_test - mu_x) @ U[:, :k]
    Yp = (Y_test - mu_y) @ V[:, :k]
    pi, _ = gw_entropic(cost_matrix(Xp), cost_matrix(Yp))
    return pi


def method_ccap_cos(X_train, Y_train, X_test, Y_test, k, **kw):
    """CCA proper (ridge-whitened) + cosine retrieval."""
    A, B, rho = fit_cca(X_train, Y_train)
    mu_x, mu_y = X_train.mean(0), Y_train.mean(0)
    Xp = (X_test - mu_x) @ A[:, :k]
    Yp = (Y_test - mu_y) @ B[:, :k]
    return cosine_scores(Xp, Yp)


def method_ccap_gw(X_train, Y_train, X_test, Y_test, k, **kw):
    """CCA proper (ridge-whitened) + entropic GW."""
    A, B, rho = fit_cca(X_train, Y_train)
    mu_x, mu_y = X_train.mean(0), Y_train.mean(0)
    Xp = (X_test - mu_x) @ A[:, :k]
    Yp = (Y_test - mu_y) @ B[:, :k]
    pi, _ = gw_entropic(cost_matrix(Xp), cost_matrix(Yp))
    return pi


# ----- Truly unsupervised methods (no audio-vision pair info at fit time) -----

def method_pca_gw(X_train, Y_train, X_test, Y_test, k, **kw):
    """Independent PCA per modality on train, then GW on projected test."""
    Vx, mu_x = fit_pca(X_train, k)
    Vy, mu_y = fit_pca(Y_train, k)
    Xp = (X_test - mu_x) @ Vx
    Yp = (Y_test - mu_y) @ Vy
    pi, _ = gw_entropic(cost_matrix(Xp), cost_matrix(Yp))
    return pi


def method_wprocrustes(X_train, Y_train, X_test, Y_test, k, **kw):
    """PCA-reduce both modalities to k dims, learn rotation via Wasserstein
    Procrustes on train (without using pair info), apply to test, cosine retrieve.
    """
    Vx, mu_x = fit_pca(X_train, k)
    Vy, mu_y = fit_pca(Y_train, k)
    Xp_tr = (X_train - mu_x) @ Vx
    Yp_tr = (Y_train - mu_y) @ Vy
    R = fit_wprocrustes(Xp_tr, Yp_tr)
    Xp_te = ((X_test - mu_x) @ Vx) @ R
    Yp_te = (Y_test - mu_y) @ Vy
    return cosine_scores(Xp_te, Yp_te)


def method_spectral_gw(X_train, Y_train, X_test, Y_test, k, **kw):
    """Transductive spectral embedding per modality (kernel on within-modality
    cosine distances), then GW on the test rows. Uses no pair info.
    """
    X_full = np.vstack([X_train, X_test])
    Y_full = np.vstack([Y_train, Y_test])
    Sx = spectral_embed(X_full, k)
    Sy = spectral_embed(Y_full, k)
    n_tr = len(X_train)
    Xp = Sx[n_tr:, :]
    Yp = Sy[n_tr:, :]
    pi, _ = gw_entropic(cost_matrix(Xp), cost_matrix(Yp))
    return pi


def method_spectral_gw_mr(X_train, Y_train, X_test, Y_test, k,
                           n_restarts=20, seed=SEED, **kw):
    """Spectral GW with multi-restart: try several random initial couplings G0,
    keep the one yielding the lowest GW cost (an unsupervised criterion).

    Addresses local minima in the non-convex GW objective. The default init
    (uniform) is included as one of the candidates.
    """
    X_full = np.vstack([X_train, X_test])
    Y_full = np.vstack([Y_train, Y_test])
    Sx = spectral_embed(X_full, k)
    Sy = spectral_embed(Y_full, k)
    n_tr = len(X_train)
    Xp = Sx[n_tr:, :]
    Yp = Sy[n_tr:, :]
    Cx = cost_matrix(Xp)
    Cy = cost_matrix(Yp)
    n = len(Xp)
    p = np.full(n, 1.0 / n)
    q = np.full(n, 1.0 / n)

    best_pi, best_cost = None, np.inf
    # default uniform init
    try:
        pi, cost = gw_entropic(Cx, Cy)
        best_pi, best_cost = pi, cost
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        G0 = rng.dirichlet(np.ones(n), size=n)
        G0 = G0 / G0.sum()
        try:
            pi, log = ot.gromov.entropic_gromov_wasserstein(
                Cx, Cy, p, q, loss_fun="square_loss",
                epsilon=EPS_ENTROPIC, max_iter=500, log=True, G0=G0,
            )
            c = float(log["gw_dist"])
            if c < best_cost:
                best_cost, best_pi = c, pi
        except Exception:
            continue
    return best_pi


# ----- Method registry, tagged by supervision level -----

# (function, supervision_tag). Supervision: "none" = no audio↔vision pair info
# at fit time; "paired" = uses (audio_i, vision_i) correspondences but no labels.
METHODS_NOK = {
    "plain_gw":   (method_plain_gw,   "none"),
    "procrustes": (method_procrustes, "paired"),
}
METHODS_K = {
    "pca_gw":         (method_pca_gw,         "none"),
    "wprocrustes":    (method_wprocrustes,    "none"),
    "spectral_gw":    (method_spectral_gw,    "none"),
    "spectral_gw_mr": (method_spectral_gw_mr, "none"),  # multi-restart variant
    "cca_cos":        (method_cca_cos,        "paired"),
    "cca_gw":         (method_cca_gw,         "paired"),
    "ccap_cos":       (method_ccap_cos,       "paired"),
    "ccap_gw":        (method_ccap_gw,        "paired"),
}


# ---------- evaluation ----------

def eval_scores(S, labels_test):
    """Compute the metric bundle from a score matrix S where S[i,j] = audio_i↔vision_j."""
    return {
        "class_purity":     class_purity(S, labels_test, labels_test),
        "hungarian_purity": hungarian_purity(S, labels_test, labels_test),
        "recall_at_5":      recall_at_k(S, labels_test, labels_test, 5),
        "recall_at_10":     recall_at_k(S, labels_test, labels_test, 10),
        "instance_r_at_1":  instance_recall_at_k(S, 1),
        "instance_r_at_10": instance_recall_at_k(S, 10),
    }


# ---------- visualization (kept as side figures) ----------

def plot_macro_clusters(X, Y, labels, vname, aname, k=K_HEADLINE) -> None:
    """t-SNE + top-2 scatter of CCA-projected embeddings. SVD fit on full data
    here (this is a viz, not a metric)."""
    U, S, V = cross_cov_svd(X, Y)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Xp = Xc @ U[:, :k]
    Yp = Yc @ V[:, :k]
    classes = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i / max(len(classes) - 1, 1)) for i, c in enumerate(classes)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, P, modality in [(axes[0, 0], Xp, f"vision · {vname}"),
                             (axes[0, 1], Yp, f"audio · {aname}")]:
        for c in classes:
            mask = labels == c
            ax.scatter(P[mask, 0], P[mask, 1], s=28, alpha=0.8, color=colors[c], label=c)
        ax.set_title(f"top-2 shared components — {modality}")
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
        ax.grid(True, alpha=0.3)
    for ax, P, modality in [(axes[1, 0], Xp, f"vision · {vname}"),
                             (axes[1, 1], Yp, f"audio · {aname}")]:
        try:
            T = TSNE(n_components=2, random_state=SEED,
                     perplexity=min(30, max(2, len(P) // 3))).fit_transform(P)
        except Exception as e:
            ax.text(0.5, 0.5, f"t-SNE failed: {e}", ha="center", va="center")
            continue
        for c in classes:
            mask = labels == c
            ax.scatter(T[mask, 0], T[mask, 1], s=28, alpha=0.8, color=colors[c], label=c)
        ax.set_title(f"t-SNE on top-{k} shared components — {modality}")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.suptitle(f"{vname}  ×  {aname}   |  shared subspace, k={k}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / f"macro_{vname}_x_{aname}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_singular_decay(decays: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, S in decays.items():
        cum = np.cumsum(S.astype(np.float64) ** 2); cum /= cum[-1]
        ax.plot(np.arange(1, len(cum) + 1), cum, label=label, alpha=0.8)
    ax.axvline(11, linestyle="--", alpha=0.4, color="grey", label="k=11 (n_class)")
    ax.axvline(5, linestyle=":", alpha=0.4, color="red", label="k=5 (sweet spot)")
    ax.set_xscale("log"); ax.set_xlabel("k (top components)")
    ax.set_ylabel("cumulative cross-covariance variance")
    ax.set_title("Singular value decay of cross-covariance (fit on train split)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / "singular_decay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- driver ----------

def chance_recall_at_k(n_test_per_class, n_test_total, k):
    """Probability that k random draws (without replacement) contain >=1 same-class item."""
    same = n_test_per_class            # number of same-class items in the test database
    other = n_test_total - same
    if k > other:
        return 1.0
    p_none = 1.0
    for i in range(k):
        p_none *= (other - i) / (n_test_total - i)
    return float(1.0 - p_none)


def main() -> None:
    aligned, _ = load_and_align(["vision", "audio"])
    vision, audio = aligned["vision"], aligned["audio"]
    if not vision or not audio:
        raise SystemExit("no embeddings found; run encode.py first")

    rows = []
    decays = {}
    first = True
    for (vname, vd), (aname, ad) in product(vision.items(), audio.items()):
        X, Y, labels = vd["X"], ad["X"], vd["labels"]
        tr, te = stratified_split(labels, TEST_FRAC, SEED)
        X_tr, Y_tr, lab_tr = X[tr], Y[tr], labels[tr]
        X_te, Y_te, lab_te = X[te], Y[te], labels[te]

        # Print chance baselines once (they depend only on test split size, same for all pairs).
        if first:
            n_test = len(te)
            per_class = n_test // 11
            print(f"test set size: {n_test}  (~{per_class}/class)")
            print(f"chance r@1={per_class/n_test:.3f}  "
                  f"r@5={chance_recall_at_k(per_class, n_test, 5):.3f}  "
                  f"r@10={chance_recall_at_k(per_class, n_test, 10):.3f}")
            first = False

        # singular value decay (fit on train only)
        U, S, V = cross_cov_svd(X_tr, Y_tr)
        decays[f"{vname} × {aname}"] = S

        print(f"\n{vname:15s} × {aname:13s}   "
              f"var(top-5,train)={variance_captured(S,5):.3f}   "
              f"var(top-11,train)={variance_captured(S,11):.3f}")

        # No-k methods
        for mname, (mfn, sup) in METHODS_NOK.items():
            S_scores = mfn(X_tr, Y_tr, X_te, Y_te)
            metrics = eval_scores(S_scores, lab_te)
            rows.append({"vision": vname, "audio": aname,
                         "method": mname, "supervision": sup, "k": None, **metrics})
            print(f"  [{sup:6s}] {mname:13s}        "
                  f"hung={metrics['hungarian_purity']:.3f}  "
                  f"r@5={metrics['recall_at_5']:.3f}  "
                  f"r@10={metrics['recall_at_10']:.3f}  "
                  f"i@1={metrics['instance_r_at_1']:.3f}")

        # k-sweep methods
        for k in K_VALUES:
            if k > min(U.shape[1], V.shape[1]):
                continue
            for mname, (mfn, sup) in METHODS_K.items():
                S_scores = mfn(X_tr, Y_tr, X_te, Y_te, k=k)
                metrics = eval_scores(S_scores, lab_te)
                row = {"vision": vname, "audio": aname,
                       "method": mname, "supervision": sup, "k": k, **metrics}
                if mname in ("cca_cos", "cca_gw"):
                    row["var_captured_train"] = variance_captured(S, k)
                rows.append(row)
                tag = f"{mname} k={k}"
                print(f"  [{sup:6s}] {tag:17s}    "
                      f"hung={metrics['hungarian_purity']:.3f}  "
                      f"r@5={metrics['recall_at_5']:.3f}  "
                      f"r@10={metrics['recall_at_10']:.3f}  "
                      f"i@1={metrics['instance_r_at_1']:.3f}")

        # Side figure: macro-cluster viz for non-music pairs
        if "music" not in aname.lower():
            plot_macro_clusters(X, Y, labels, vname, aname, k=K_HEADLINE)

    plot_singular_decay(decays)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
