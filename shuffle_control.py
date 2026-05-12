"""Sanity check: are the methods using the data, or the manifest row order?

The cache/embeds/*.npz files are sorted by instrument (groups of 22 same-class
samples). If a decomposition collapses to a data-blind output (e.g., KPCA RBF
with median bandwidth on concentrated high-D embeddings), the resulting
"alignment" is a trivial identity mapping that exploits the shared row ordering
of the two modalities, not any cross-modal signal.

This script shuffles vision rows + vision labels jointly (audio unchanged),
re-runs each method on a few (vision, audio) pairs, and compares recall@k
before and after.

Interpretation:
  - A *data-using* method: recall stays roughly the same. Cluster geometry is
    invariant to row permutation, so GW still aligns same-class clusters across
    modalities and class-level recall@k is preserved.
  - A *data-blind* method: recall crashes to chance. Its 'alignment' was the
    shared row order, which the shuffle has destroyed.

If you see kpca_rbf collapse here, the bandwidth heuristic (now local k-NN
median, see comparison_matrix.fit_kpca_rbf) didn't escape the degeneracy.
"""
import numpy as np

from analyze import load_and_align, recall_at_k
from comparison_matrix import (
    m_vanilla,
    m_svd_truncate,
    m_spectral_whiten,
    m_kpca_rbf,
    m_spectral_gw_mr,
    m_wprocrustes,
    distance_correlation,
)

SEED = 42

METHODS = {
    "vanilla":         (m_vanilla,         None),
    "svd_truncate":    (m_svd_truncate,    100),
    "spectral_whiten": (m_spectral_whiten, 100),
    "kpca_rbf":        (m_kpca_rbf,        100),
    "spectral_gw_mr":  (m_spectral_gw_mr,  100),
    "wprocrustes":     (m_wprocrustes,     100),
}

# Two suspicious pairs where kpca_rbf k=5 hit r@11=0.975 with dcor=1.0,
# plus one pair that produced a legitimate top result.
SUSPICIOUS = [
    ("clip-large",   "mert-330m"),
    ("dinov2-base",  "mert-95m"),
]
CONTROL = [
    ("dinov2-small", "clap-unfused"),
]


def run_pair(tag, vname, aname, Xv, Xa, labels):
    print(f"\n=== [{tag}] {vname} × {aname} ===")
    print(f"  {'method':<17}{'k':<4}{'r@5 unsh→shuf':<22}{'r@11 unsh→shuf':<22}"
          f"{'dcor unsh/shuf':<16}flag")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xv))
    Xv_s = Xv[perm]
    labels_v_s = labels[perm]

    for mname, (mfn, k) in METHODS.items():
        kw = {} if k is None else {"k": k}
        try:
            M0, _, Cv0, Ca0 = mfn(Xv, Xa, **kw)
            r5_0  = recall_at_k(M0, labels,    labels, 5)
            r11_0 = recall_at_k(M0, labels,    labels, 11)
            d0    = distance_correlation(Cv0, Ca0)
        except Exception as e:
            print(f"  {mname:<17}{str(k):<4}  unshuffled FAILED: {e.__class__.__name__}: {e}")
            continue
        try:
            M1, _, Cv1, Ca1 = mfn(Xv_s, Xa, **kw)
            r5_1  = recall_at_k(M1, labels_v_s, labels, 5)
            r11_1 = recall_at_k(M1, labels_v_s, labels, 11)
            d1    = distance_correlation(Cv1, Ca1)
        except Exception as e:
            print(f"  {mname:<17}{str(k):<4}  shuffled FAILED: {e.__class__.__name__}: {e}")
            continue

        flag = ""
        if d0 > 0.95:
            flag = "DEGENERATE"
        elif r5_0 > 0.2 and r5_1 < 0.4 * r5_0:
            flag = "data-blind (recall crashed)"
        elif abs(r5_1 - r5_0) < 0.05 and r5_0 > 0.3:
            flag = "data-using ✓"

        print(f"  {mname:<17}{str(k):<4}"
              f"{r5_0:.3f} → {r5_1:.3f} ({r5_1 - r5_0:+.3f})   "
              f"{r11_0:.3f} → {r11_1:.3f} ({r11_1 - r11_0:+.3f})   "
              f"{d0:.2f} / {d1:.2f}     {flag}")


def main():
    aligned, _ = load_and_align(["vision", "audio"])
    vision = aligned["vision"]; audio = aligned["audio"]
    if not vision or not audio:
        raise SystemExit("no embeddings found; run encode.py first")
    labels = next(iter(vision.values()))["labels"]

    for tag, pairs in [("SUSPICIOUS", SUSPICIOUS), ("CONTROL", CONTROL)]:
        for vname, aname in pairs:
            if vname not in vision or aname not in audio:
                print(f"!! skip {vname} × {aname} (encoder not in cache)")
                continue
            run_pair(tag, vname, aname, vision[vname]["X"], audio[aname]["X"], labels)


if __name__ == "__main__":
    main()
