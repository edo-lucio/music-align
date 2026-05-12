"""Unified decomposition + GW comparison matrix.

Most methods are unsupervised: they use neither audio↔vision pair correspondences
nor class labels at fit time. CCA is a paired baseline (uses row-pairing at fit
time, no class labels) for comparison against the unsupervised methods. Each
method produces a square score / coupling matrix M, evaluated by class_purity,
hungarian_purity, recall@{5,11,22}, instance_r@{1,10}, FOSCTTM, MRR, NMI, GW
cost residual, post-decomp dCor(Cv, Ca), and CKA(1-Cv, 1-Ca).

Methods:
  NOK (no k):
    vanilla         — raw embeddings → entropic GW
    random_proj     — Gaussian projection to k_JL → entropic GW
  K-sweep:
    svd_truncate    — top-k PCA per modality → entropic GW
    spectral_whiten — top-k PCA, singular values normalized to 1 → entropic GW
    kpca_rbf        — KPCA(RBF) per modality → entropic GW
    wprocrustes     — top-k PCA + Wasserstein-Procrustes rotation; cosine score
    spectral_gw     — normalized-Laplacian RBF eigenmap → entropic GW
    spectral_gw_mr  — spectral_gw with multi-restart (lowest GW cost wins)
    cca             — Canonical Correlation Analysis into shared k-dim space;
                      cosine score (PAIRED supervision)

Writes comparison_matrix.json as a flat list of rows; schema matches what
report.py consumes.
"""
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import ot
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import normalized_mutual_info_score
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

OUT = Path("comparison_matrix.json")
SEED = 0
K_VALUES = [100]
JL_EPS = 0.5            # k_JL = ceil(8 ln(N) / eps^2)
EPS_ENTROPIC = 0.005     # matches analyze.gw_entropic
N_RESTARTS = 20         # for spectral_gw_mr
DEGEN_DCOR = 0.95       # dcor(Cv, Ca) > this => decomposition collapsed; drop row


def jl_k(n, eps=JL_EPS):
    return int(np.ceil(8.0 * np.log(n) / (eps ** 2)))


def fit_svd_truncate(X, k):
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] * S[:k]  # PCA scores


def fit_whiten(X, k):
    """Top-k left singular vectors with magnitudes removed (unit-variance cols)."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] * np.sqrt(len(X))


def fit_random_proj(X, k, seed):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    if k >= D:
        return X.copy()
    R = rng.standard_normal((D, k)) / np.sqrt(k)
    return X @ R


KPCA_KNN = 7  # bandwidth = median of k-th nearest-neighbor distance


def _local_bandwidth(D, knn=KPCA_KNN):
    """Median of knn-th nearest-neighbor distance.

    Global-median bandwidth degenerates on high-D embeddings where cosine
    distances concentrate around a constant — the kernel becomes near α·J+(1-α)·I,
    KernelPCA returns data-blind eigenvectors of the constant subspace, and
    distinct inputs produce identical outputs. A local bandwidth (driven by each
    point's own neighborhood) avoids this collapse.
    """
    D_self_inf = D.copy()
    np.fill_diagonal(D_self_inf, np.inf)
    knn_dist = np.partition(D_self_inf, knn - 1, axis=1)[:, :knn].max(axis=1)
    return max(float(np.median(knn_dist)), 1e-12)


def fit_kpca_rbf(X, k):
    D = cosine_distances(X)
    sigma = _local_bandwidth(D)
    gamma = 1.0 / (2.0 * sigma ** 2)
    kpca = KernelPCA(n_components=k, kernel="rbf", gamma=gamma, random_state=SEED)
    return kpca.fit_transform(X)


def fit_spectral_embed(X, k):
    """Normalized-Laplacian eigenmap on an RBF kernel over cosine distances."""
    D = cosine_distances(X)
    sigma = max(np.median(D[D > 0]), 1e-12)
    K = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(K.sum(axis=1), 1e-12))
    K_norm = K * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    _, V = np.linalg.eigh(K_norm)
    return V[:, -k:]  # eigh: ascending -> top-k from the right


def fit_cca(Xv, Xa, k):
    """Canonical Correlation Analysis: projects both modalities into a shared
    k-dim space where per-component correlation is maximized. Uses row-pairing
    at fit time -> paired supervision. D > N is acceptable to sklearn (NIPALS),
    but the result will be overfit at this N=242 since there is no held-out
    split; treat the score as an in-sample upper bound for paired alignment.
    """
    k = min(k, Xv.shape[1], Xa.shape[1], len(Xv) - 1)
    cca = CCA(n_components=k, max_iter=1000)
    return cca.fit_transform(Xv, Xa)


def fit_wprocrustes_R(X_a, X_b, n_iters=30, sinkhorn_reg=0.05, seed=SEED):
    """Wasserstein-Procrustes (Grave et al. 2018): rotation R aligning X_a to X_b
    without pair info. Sinkhorn OT + orthogonal Procrustes alternation. Both
    point clouds must share dim.
    """
    rng = np.random.default_rng(seed)
    n_a, d = X_a.shape
    n_b = len(X_b)
    R = np.linalg.qr(rng.standard_normal((d, d)))[0]
    p = np.full(n_a, 1.0 / n_a); q = np.full(n_b, 1.0 / n_b)
    for _ in range(n_iters):
        XR = X_a @ R
        sq_a = (XR ** 2).sum(1); sq_b = (X_b ** 2).sum(1)
        C = sq_a[:, None] + sq_b[None, :] - 2 * XR @ X_b.T
        C = np.maximum(C, 0.0)
        C = C / max(C.max(), 1e-12)
        P = ot.sinkhorn(p, q, C, reg=sinkhorn_reg, numItermax=200, stopThr=1e-6)
        M = X_a.T @ P @ X_b
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
    return R


def cosine_scores(A, B, eps=1e-12):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A @ B.T


# ---------- metrics ----------

def foscttm(M):
    """Fraction Of Samples Closer Than True Match. M is similarity (higher=closer);
    true match is the diagonal entry (paired data: same row index across modalities).
    Symmetrized over rows and columns. Chance ≈ 0.5, perfect = 0.
    """
    N = M.shape[0]
    fos_rows = fos_cols = 0.0
    for i in range(N):
        tr = M[i, i]
        row = M[i].copy(); row[i] = -np.inf
        fos_rows += (row > tr).sum() / (N - 1)
        col = M[:, i].copy(); col[i] = -np.inf
        fos_cols += (col > tr).sum() / (N - 1)
    return float(0.5 * (fos_rows + fos_cols) / N)


def distance_correlation(A, B):
    """Szekely distance correlation between two square distance matrices."""
    a_row = A.mean(axis=0, keepdims=True); a_col = A.mean(axis=1, keepdims=True)
    A_c = A - a_row - a_col + A.mean()
    b_row = B.mean(axis=0, keepdims=True); b_col = B.mean(axis=1, keepdims=True)
    B_c = B - b_row - b_col + B.mean()
    dCov2 = (A_c * B_c).mean()
    dVarA = (A_c * A_c).mean(); dVarB = (B_c * B_c).mean()
    if dVarA <= 0 or dVarB <= 0:
        return 0.0
    return float(np.sqrt(max(dCov2, 0.0) / np.sqrt(dVarA * dVarB)))


def chance_recall_at_k(per_class, n_total, k):
    same, other = per_class, n_total - per_class
    if k > other:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (other - i) / (n_total - i)
    return 1.0 - p


def mrr(M):
    """Mean Reciprocal Rank of the true match (diagonal). Symmetrized.
    Chance ≈ (1 + H_{N-1})/N ≈ (ln N + γ)/N for large N (about 0.023 at N=242).
    """
    N = M.shape[0]
    s = 0.0
    for i in range(N):
        tr = M[i, i]
        row = M[i].copy(); row[i] = -np.inf
        rank = 1 + int((row > tr).sum())
        s += 1.0 / rank
        tr = M[i, i]
        col = M[:, i].copy(); col[i] = -np.inf
        rank = 1 + int((col > tr).sum())
        s += 1.0 / rank
    return float(s / (2 * N))


def nmi(M, labels_a, labels_b):
    """Normalized Mutual Information between argmax assignments and labels.
    Permutation-invariant; insensitive to class-relabeling artifacts in GW.
    Chance ≈ 0 (sklearn's normalization is symmetric).
    """
    preds = labels_b[np.asarray(M).argmax(axis=1)]
    return float(normalized_mutual_info_score(labels_a, preds))


def cka_from_distance(Cv, Ca):
    """Centered Kernel Alignment using kernels K = 1 - cosine_distance.
    Permutation-invariant in the feature dimension. CKA ∈ [0, 1]; higher = the
    two distance geometries are more aligned. Stricter than dCor: HSIC weights
    high-similarity pairs more.
    """
    Kv = 1.0 - Cv
    Ka = 1.0 - Ca
    Kv_c = Kv - Kv.mean(0, keepdims=True) - Kv.mean(1, keepdims=True) + Kv.mean()
    Ka_c = Ka - Ka.mean(0, keepdims=True) - Ka.mean(1, keepdims=True) + Ka.mean()
    num = (Kv_c * Ka_c).sum()
    den2 = (Kv_c * Kv_c).sum() * (Ka_c * Ka_c).sum()
    if den2 <= 0:
        return 0.0
    return float(num / np.sqrt(den2))


# ---------- methods: each returns (M, gw_cost_or_None, Cv, Ca) ----------

def m_vanilla(Xv, Xa, **_):
    Cv, Ca = cost_matrix(Xv), cost_matrix(Xa)
    pi, c = gw_entropic(Cv, Ca)
    return pi, c, Cv, Ca


def m_random_proj(Xv, Xa, k, **_):
    Vp = fit_random_proj(Xv, k, SEED)
    Ap = fit_random_proj(Xa, k, SEED + 1)
    Cv, Ca = cost_matrix(Vp), cost_matrix(Ap)
    pi, c = gw_entropic(Cv, Ca)
    return pi, c, Cv, Ca


def m_svd_truncate(Xv, Xa, k, **_):
    Vp, Ap = fit_svd_truncate(Xv, k), fit_svd_truncate(Xa, k)
    Cv, Ca = cost_matrix(Vp), cost_matrix(Ap)
    pi, c = gw_entropic(Cv, Ca)
    return pi, c, Cv, Ca


def m_spectral_whiten(Xv, Xa, k, **_):
    Vp, Ap = fit_whiten(Xv, k), fit_whiten(Xa, k)
    Cv, Ca = cost_matrix(Vp), cost_matrix(Ap)
    pi, c = gw_entropic(Cv, Ca)
    return pi, c, Cv, Ca


def m_kpca_rbf(Xv, Xa, k, **_):
    Vp, Ap = fit_kpca_rbf(Xv, k), fit_kpca_rbf(Xa, k)
    Cv, Ca = cost_matrix(Vp), cost_matrix(Ap)
    pi, c = gw_entropic(Cv, Ca)
    return pi, c, Cv, Ca


def m_wprocrustes(Xv, Xa, k, **_):
    """PCA-reduce both, learn rotation R audio→vision via Sinkhorn-Procrustes,
    score with cosine similarity. No GW step → gw_cost is None.
    """
    Vp = fit_svd_truncate(Xv, k)
    Ap = fit_svd_truncate(Xa, k)
    R = fit_wprocrustes_R(Ap, Vp)
    Ap_rot = Ap @ R
    M = cosine_scores(Vp, Ap_rot)
    Cv = cost_matrix(Vp); Ca = cost_matrix(Ap_rot)
    return M, None, Cv, Ca


def m_spectral_gw(Xv, Xa, k, **_):
    Vp, Ap = fit_spectral_embed(Xv, k), fit_spectral_embed(Xa, k)
    Cv, Ca = cost_matrix(Vp), cost_matrix(Ap)
    pi, c = gw_entropic(Cv, Ca)
    return pi, c, Cv, Ca


def m_cca(Xv, Xa, k, **_):
    """CCA into a shared k-dim space; cosine-similarity scoring. PAIRED.
    No GW step -> gw_cost is None. Cv/Ca for the degeneracy filter only.
    """
    Vp, Ap = fit_cca(Xv, Xa, k)
    M = cosine_scores(Vp, Ap)
    Cv = cost_matrix(Vp); Ca = cost_matrix(Ap)
    return M, None, Cv, Ca


def m_spectral_gw_mr(Xv, Xa, k, n_restarts=N_RESTARTS, **_):
    """Spectral GW with multi-restart: random Dirichlet inits + uniform; keep
    the run with the lowest GW objective."""
    Vp, Ap = fit_spectral_embed(Xv, k), fit_spectral_embed(Xa, k)
    Cv, Ca = cost_matrix(Vp), cost_matrix(Ap)
    n = len(Vp)
    p = np.full(n, 1.0 / n); q = np.full(n, 1.0 / n)
    best_pi, best_cost = None, np.inf
    try:
        pi, c = gw_entropic(Cv, Ca)
        best_pi, best_cost = pi, c
    except Exception:
        pass
    rng = np.random.default_rng(SEED)
    for _ in range(n_restarts):
        G0 = rng.dirichlet(np.ones(n), size=n); G0 = G0 / G0.sum()
        try:
            pi, log = ot.gromov.entropic_gromov_wasserstein(
                Cv, Ca, p, q, loss_fun="square_loss",
                epsilon=EPS_ENTROPIC, max_iter=500, log=True, G0=G0,
            )
            c = float(log["gw_dist"])
            if c < best_cost:
                best_cost, best_pi = c, pi
        except Exception:
            continue
    return best_pi, best_cost, Cv, Ca


METHODS_NOK = {
    "vanilla": m_vanilla,
}
METHODS_K = {
    "svd_truncate":    m_svd_truncate,
    "spectral_whiten": m_spectral_whiten,
    "kpca_rbf":        m_kpca_rbf,
    "wprocrustes":     m_wprocrustes,
    "spectral_gw":     m_spectral_gw,
    "spectral_gw_mr":  m_spectral_gw_mr,
    "cca":             m_cca,
}
PAIRED_METHODS = {"cca"}


# ---------- driver ----------

def eval_M(M, labels):
    return {
        "class_purity":     class_purity(M, labels, labels),
        "hungarian_purity": hungarian_purity(M, labels, labels),
        "recall_at_5":      recall_at_k(M, labels, labels, 5),
        "recall_at_11":     recall_at_k(M, labels, labels, 11),
        "recall_at_22":     recall_at_k(M, labels, labels, 22),
        "instance_r_at_1":  instance_recall_at_k(M, 1),
        "instance_r_at_10": instance_recall_at_k(M, 10),
        "mrr":              mrr(M),
        "nmi":              nmi(M, labels, labels),
    }


def _make_row(vname, aname, mname, k, M, gw_cost, Cv, Ca, labels):
    return {
        "vision": vname, "audio": aname,
        "method": mname,
        "supervision": "paired" if mname in PAIRED_METHODS else "none",
        "k": k,
        **eval_M(M, labels),
        "foscttm":         foscttm(M),
        "gw_cost":         (None if gw_cost is None else float(gw_cost)),
        "dcor_postdecomp": distance_correlation(Cv, Ca),
        "cka":             cka_from_distance(Cv, Ca),
    }


def _print_row(row, degenerate=False):
    tag = f"{row['method']} k={row['k']}" if row["k"] is not None else row["method"]
    flag = "  ⚠ DEGENERATE (dropped)" if degenerate else ""
    print(f"  {tag:25s}  "
          f"mrr={row['mrr']:.3f}  "
          f"nmi={row['nmi']:.3f}  "
          f"cka={row['cka']:.3f}  "
          f"hung={row['hungarian_purity']:.3f}  "
          f"r@11={row['recall_at_11']:.3f}  "
          f"foscttm={row['foscttm']:.3f}  "
          f"dcor={row['dcor_postdecomp']:.3f}{flag}")


def _is_degenerate(row):
    """A decomposition has collapsed if vision and audio post-decomp distance
    matrices are nearly identical: dcor(Cv, Ca) close to 1.0. The 'alignment'
    is then a trivial artifact of the (shared) manifest row ordering, not a
    cross-modal signal. Filter these rows out before reporting.
    """
    return row["dcor_postdecomp"] > DEGEN_DCOR


def main():
    aligned, _ = load_and_align(["vision", "audio"])
    vision, audio = aligned["vision"], aligned["audio"]
    if not vision or not audio:
        raise SystemExit("no embeddings found; run encode.py first")

    labels = next(iter(vision.values()))["labels"]
    N = len(labels)
    per_class = max(Counter(labels.tolist()).values())
    k_JL = jl_k(N, JL_EPS)
    print(f"N={N}  per-class={per_class}  k_JL(eps={JL_EPS})={k_JL}")
    print(f"chance r@5={chance_recall_at_k(per_class, N, 5):.3f}  "
          f"r@11={chance_recall_at_k(per_class, N, 11):.3f}  "
          f"r@22={chance_recall_at_k(per_class, N, 22):.3f}  "
          f"foscttm~0.5  instance r@1={1/N:.4f}")

    rows = []
    for (vname, vd), (aname, ad) in product(vision.items(), audio.items()):
        Xv, Xa = vd["X"], ad["X"]
        print(f"\n== {vname:15s} x {aname:13s} ==")

        for mname, mfn in METHODS_NOK.items():
            try:
                M, gw_cost, Cv, Ca = mfn(Xv, Xa)
            except Exception as e:
                print(f"  {mname:17s} FAILED: {e.__class__.__name__}: {e}")
                continue
            row = _make_row(vname, aname, mname, None, M, gw_cost, Cv, Ca, labels)
            if _is_degenerate(row):
                _print_row(row, degenerate=True); continue
            rows.append(row); _print_row(row)

        try:
            M, gw_cost, Cv, Ca = m_random_proj(Xv, Xa, k=k_JL)
            row = _make_row(vname, aname, "random_proj", k_JL, M, gw_cost, Cv, Ca, labels)
            if _is_degenerate(row):
                _print_row(row, degenerate=True)
            else:
                rows.append(row); _print_row(row)
        except Exception as e:
            print(f"  random_proj k={k_JL:<3d} FAILED: {e.__class__.__name__}: {e}")

        for k in K_VALUES:
            for mname, mfn in METHODS_K.items():
                try:
                    M, gw_cost, Cv, Ca = mfn(Xv, Xa, k=k)
                except Exception as e:
                    print(f"  {mname:17s} k={k:<3d} FAILED: {e.__class__.__name__}: {e}")
                    continue
                row = _make_row(vname, aname, mname, k, M, gw_cost, Cv, Ca, labels)
                if _is_degenerate(row):
                    _print_row(row, degenerate=True); continue
                rows.append(row); _print_row(row)

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
