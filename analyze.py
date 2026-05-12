"""Run GW variants between vision and audio embeddings; report metrics.

Metrics per (vision_model, audio_model, gw_variant):
    - class_purity:  fraction of points whose argmax-matched partner shares the instrument label
    - gw_cost:       optimization value of the GW objective
    - mutual_knn:    intra-modal neighborhood overlap (alignment-free, Platonic-style)

Results saved to results.json and a CSV table printed to stdout.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import ot
import ot.gromov as gw
from sklearn.metrics.pairwise import cosine_distances

EMBEDS = Path("cache/embeds")
RESULTS = Path("results.json")
EPS_ENTROPIC = 0.05
KNN_K = 10
SLICED_PROJ = 100
SEED = 0
FGW_ALPHAS = [0.05, 0.1, 0.2]  # text-bias weight; small = mostly GW, light supervision dose


# ---------- cost / distance ----------

def cost_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine distance matrix, normalized to [0,1] range for stable GW."""
    D = cosine_distances(X).astype(np.float64)
    D /= max(D.max(), 1e-12)
    return D


# ---------- GW variants ----------

def gw_entropic(Ca, Cb):
    p = np.full(len(Ca), 1.0 / len(Ca))
    q = np.full(len(Cb), 1.0 / len(Cb))
    pi, log = gw.entropic_gromov_wasserstein(
        Ca, Cb, p, q, loss_fun="square_loss",
        epsilon=EPS_ENTROPIC, max_iter=500, log=True,
    )
    return pi, float(log["gw_dist"])


def gw_unbalanced(Ca, Cb):
    """Pure unbalanced GW: FUGW with the fused linear term zeroed out."""
    p = np.full(len(Ca), 1.0 / len(Ca))
    q = np.full(len(Cb), 1.0 / len(Cb))
    pi, _duals, log = gw.fused_unbalanced_gromov_wasserstein(
        Ca, Cb, wx=p, wy=q,
        reg_marginals=1.0, epsilon=EPS_ENTROPIC,
        divergence="kl", alpha=0.0, M=None,
        max_iter=200, log=True,
    )
    return pi, float(log["fugw_cost"])


def gw_sliced(X, Y, n_proj=SLICED_PROJ, seed=SEED):
    """Sliced GW (Vayer et al. 2019): 1D random projections + sorted/reverse pairing.

    Returns (None, cost) — sliced GW does not yield a coupling matrix.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    costs = []
    for _ in range(n_proj):
        tx = rng.standard_normal(X.shape[1]); tx /= np.linalg.norm(tx) + 1e-12
        ty = rng.standard_normal(Y.shape[1]); ty /= np.linalg.norm(ty) + 1e-12
        xs = np.sort(X @ tx)
        ys = np.sort(Y @ ty)
        Dx = np.abs(xs[:, None] - xs[None, :])
        Dy = np.abs(ys[:, None] - ys[None, :])
        c_id = ((Dx - Dy) ** 2).mean()
        c_rev = ((Dx - Dy[::-1, ::-1]) ** 2).mean()
        costs.append(min(c_id, c_rev))
    return None, float(np.mean(costs))


def gw_fused(Cv, Ca, M, alpha_text):
    """Fused GW. Our convention: (1 - alpha_text)*GW + alpha_text*<M, pi>.

    POT's `alpha` weights the structural (GW) term, so we pass 1 - alpha_text.
    Small alpha_text -> mostly structure, light text/label nudge.
    """
    n = len(Cv)
    p = np.full(n, 1.0 / n)
    q = np.full(n, 1.0 / n)
    pi, log = gw.entropic_fused_gromov_wasserstein(
        M, Cv, Ca, p, q, loss_fun="square_loss",
        epsilon=EPS_ENTROPIC, alpha=1.0 - alpha_text,
        max_iter=500, log=True,
    )
    return pi, float(log["fgw_dist"])


def build_M_text(text_X):
    """Cosine-distance cost matrix from text embeddings (same shape & scale as Cv/Ca)."""
    return cost_matrix(text_X)


def build_M_label(labels_a, labels_b):
    """Hard-label cost: 0 inside same-class block, 1 elsewhere."""
    return (np.asarray(labels_a)[:, None] != np.asarray(labels_b)[None, :]).astype(np.float64)


def build_M_shuffled_text(text_X, seed):
    """Negative control: shuffle column correspondence to break the M[i,i]=0 hint.

    Previously permuting the underlying text kept the diagonal at zero (since
    cos_dist(text[s[i]], text[s[i]]) = 0), which is a free identity match.
    Shuffling columns gives M_shuf[i,i] = cos_dist(text[i], text[s[i]]) > 0,
    actually removing both off-diagonal class structure AND the identity hint.
    """
    M = cost_matrix(text_X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M.shape[1])
    return M[:, perm]


def build_M_identity(n):
    """Identity-pull-only baseline: 0 on the diagonal, 1 elsewhere.

    Measures how much FGW purity comes from the trivial M[i,i]=0 hint alone,
    independent of any semantic content in M. Lift of fgw[text]/fgw[label]
    over fgw[identity] is the actual semantic contribution.
    """
    return 1.0 - np.eye(n, dtype=np.float64)


def mask_diagonal(M):
    """Replace M's diagonal with each row's off-diagonal mean.

    Removes the trivial M[i,i]=0 hint that lets FGW recover identity matches
    for free on paired data. After masking, FGW must rely on off-diagonal
    structure (i.e. the actual semantic content of M).
    """
    M = np.asarray(M, dtype=np.float64).copy()
    n = M.shape[0]
    off_sum = M.sum(axis=1) - np.diag(M)
    off_mean = off_sum / max(n - 1, 1)
    np.fill_diagonal(M, off_mean)
    return M


# ---------- metrics ----------

def class_purity(pi: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    j = pi.argmax(axis=1)
    return float(np.mean(labels_a == labels_b[j]))


def hungarian_purity(pi: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Class-permutation-invariant purity.

    Builds an (n_class x n_class) soft confusion matrix from pi, finds the
    optimal class permutation, then reports argmax accuracy under it.
    Robust to GW's invariance to global class relabelling.
    """
    from scipy.optimize import linear_sum_assignment
    classes = sorted(set(labels_a.tolist()) | set(labels_b.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    ia = np.array([idx[c] for c in labels_a])
    ib = np.array([idx[c] for c in labels_b])
    M = np.zeros((len(classes), len(classes)))
    np.add.at(M, (ia[:, None].repeat(len(labels_b), axis=1),
                  ib[None, :].repeat(len(labels_a), axis=0)), pi)
    row, col = linear_sum_assignment(-M)
    perm = dict(zip(row, col))  # source-class-idx -> target-class-idx
    j = pi.argmax(axis=1)
    return float(np.mean([ib[j[i]] == perm[ia[i]] for i in range(len(labels_a))]))


def recall_at_k(pi: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> float:
    """Class-level recall@k: top-k targets include any same-instrument item."""
    topk = np.argpartition(-pi, kth=min(k, pi.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean([
        labels_a[i] in labels_b[topk[i]] for i in range(len(labels_a))
    ]))


def instance_recall_at_k(pi: np.ndarray, k: int) -> float:
    """Instance-level recall@k: query's own paired partner is in the top-k.

    Chance: k / N (e.g. 0.004 for k=1, ~0.041 for k=10 with N=242). Read any
    value below ~0.03 as noise even if it's nominally above chance.
    """
    topk = np.argpartition(-pi, kth=min(k, pi.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean([i in topk[i] for i in range(pi.shape[0])]))


def mutual_knn(Ca: np.ndarray, Cb: np.ndarray, k: int = KNN_K) -> float:
    """Average overlap of k-NN sets between the two cost matrices (Platonic-style)."""
    nbr_a = np.argsort(Ca, axis=1)[:, 1 : k + 1]
    nbr_b = np.argsort(Cb, axis=1)[:, 1 : k + 1]
    return float(np.mean([
        len(set(a) & set(b)) / k for a, b in zip(nbr_a, nbr_b)
    ]))


# ---------- driver ----------

def _load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"]).squeeze()
    if X.ndim != 2:
        raise SystemExit(
            f"{path.name}: expected 2D embeddings (N, D), got shape {d['X'].shape}. "
            "Re-run encode.py after pulling the fix."
        )
    return {"X": X, "labels": np.asarray(d["labels"]), "ids": np.asarray(d["ids"])}


def load_and_align(prefixes: list[str]) -> tuple[dict[str, dict[str, dict]], np.ndarray]:
    """Load all npz under each prefix, align them to the common id intersection.

    Returns ({prefix: {name: {X, labels, ids}}}, common_ids) where every X is
    reordered to match common_ids.
    """
    raw: dict[str, dict[str, dict]] = {}
    id_sets = []
    for pre in prefixes:
        raw[pre] = {}
        for f in sorted(EMBEDS.glob(f"{pre}_*.npz")):
            d = _load_npz(f)
            raw[pre][f.stem.replace(f"{pre}_", "")] = d
            id_sets.append(set(d["ids"].tolist()))
    if not id_sets:
        return raw, np.array([])
    common = sorted(set.intersection(*id_sets))
    common_arr = np.array(common)
    for pre, mods in raw.items():
        for name, d in mods.items():
            id_to_idx = {i: k for k, i in enumerate(d["ids"].tolist())}
            order = np.array([id_to_idx[i] for i in common])
            d["X"] = d["X"][order]
            d["labels"] = d["labels"][order]
            d["ids"] = d["ids"][order]
            if len(d["ids"]) < max(len(s) for s in id_sets):
                print(f"  note: {pre}/{name} aligned to {len(common)} common ids "
                      f"(had {len(id_to_idx)})")
    return raw, common_arr


def _score(vname, aname, variant, pi, cost, mk, labels):
    nan = float("nan")
    if pi is None:
        return {"vision": vname, "audio": aname, "variant": variant,
                "gw_cost": cost, "class_purity": nan, "hungarian_purity": nan,
                "recall_at_5": nan, "recall_at_10": nan,
                "instance_r_at_1": nan, "instance_r_at_10": nan,
                "mutual_knn": mk}
    return {
        "vision": vname, "audio": aname, "variant": variant, "gw_cost": cost,
        "class_purity": class_purity(pi, labels, labels),
        "hungarian_purity": hungarian_purity(pi, labels, labels),
        "recall_at_5": recall_at_k(pi, labels, labels, 5),
        "recall_at_10": recall_at_k(pi, labels, labels, 10),
        "instance_r_at_1": instance_recall_at_k(pi, 1),
        "instance_r_at_10": instance_recall_at_k(pi, 10),
        "mutual_knn": mk,
    }


def _print_row(r):
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and not np.isnan(x) else "  nan"
    print(f"{r['vision']:15s} {r['audio']:13s} {r['variant']:18s}  "
          f"pur={fmt(r['class_purity'])}  hung={fmt(r['hungarian_purity'])}  "
          f"r@5={fmt(r['recall_at_5'])}  r@10={fmt(r['recall_at_10'])}  "
          f"i@1={fmt(r['instance_r_at_1'])}  i@10={fmt(r['instance_r_at_10'])}  "
          f"cost={fmt(r['gw_cost'])}  mknn={fmt(r['mutual_knn'])}")


def compose_bridge(pi_AT: np.ndarray, pi_VT: np.ndarray) -> np.ndarray:
    """Compose A->T and V->T couplings into an induced A->V coupling.

    Both couplings have uniform marginals q_T = 1/N. The Markov composition
    pi_AV[i,j] = sum_k pi_AT[i,k] * pi_VT[j,k] / q_T[k]
    preserves the A marginal so argmax over j is a valid match.
    """
    n_t = pi_AT.shape[1]
    return n_t * (pi_AT @ pi_VT.T)


def main() -> None:
    aligned, common = load_and_align(["vision", "audio", "text"])
    vision, audio, text = aligned["vision"], aligned["audio"], aligned["text"]
    if not vision or not audio:
        raise SystemExit("no embeddings found; run encode.py first")
    print(f"aligned to {len(common)} common items across modalities")

    rows = []
    for (vname, vd), (aname, ad) in product(vision.items(), audio.items()):
        Cv = cost_matrix(vd["X"])
        Ca = cost_matrix(ad["X"])
        labels = vd["labels"]
        mk = mutual_knn(Cv, Ca)

        for variant, fn in [
            ("entropic",   lambda: gw_entropic(Cv, Ca)),
            ("unbalanced", lambda: gw_unbalanced(Cv, Ca)),
            ("sliced",     lambda: gw_sliced(vd["X"], ad["X"])),
        ]:
            try:
                pi, cost = fn()
            except Exception as e:
                pi, cost = None, float("nan")
                print(f"  ! {variant} failed for ({vname},{aname}): {e}")
            rows.append(_score(vname, aname, variant, pi, cost, mk, labels))
            _print_row(rows[-1])

        # ---- Bridge through text (Approach 1): A->T->V via two GWs ----
        for tname, td in text.items():
            Ct = cost_matrix(td["X"])
            try:
                pi_AT, cost_AT = gw_entropic(Ca, Ct)
                pi_VT, cost_VT = gw_entropic(Cv, Ct)
                pi_bridge = compose_bridge(pi_AT, pi_VT)
                cost_b = cost_AT + cost_VT
            except Exception as e:
                pi_bridge, cost_b = None, float("nan")
                print(f"  ! bridge[{tname}] failed for ({vname},{aname}): {e}")
            rows.append(_score(vname, aname, f"bridge[{tname}]",
                               pi_bridge, cost_b, mk, labels))
            _print_row(rows[-1])

        # ---- Fused GW: supervision-dose sweep with text / label / shuffled / identity M ----
        # Diagonals are masked (replaced with off-diagonal row mean) so FGW can't
        # exploit M[i,i]=0 as a free identity match. fgw[identity] is kept UNMASKED
        # as the trivialization baseline: it should still saturate high.
        M_label = mask_diagonal(build_M_label(labels, labels))
        M_ident = build_M_identity(len(labels))  # NOT masked: this IS the diagonal baseline
        text_for_fgw = next(iter(text.values()), None)
        M_text = mask_diagonal(build_M_text(text_for_fgw["X"])) if text_for_fgw is not None else None
        M_shuf = mask_diagonal(build_M_shuffled_text(text_for_fgw["X"], SEED)) if text_for_fgw is not None else None
        for atext in FGW_ALPHAS:
            for mtag, M in [("label", M_label), ("text", M_text),
                            ("shuffled", M_shuf), ("identity", M_ident)]:
                if M is None:
                    continue
                try:
                    pi_f, cost_f = gw_fused(Cv, Ca, M, atext)
                except Exception as e:
                    pi_f, cost_f = None, float("nan")
                    print(f"  ! fgw[{mtag},a={atext}] failed for ({vname},{aname}): {e}")
                rows.append(_score(vname, aname,
                                   f"fgw[{mtag},a={atext}]",
                                   pi_f, cost_f, mk, labels))
                _print_row(rows[-1])

    RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()
