#!/usr/bin/env python3
"""
================================================================================
END-TO-END CLUSTERING PIPELINE: SWEEP + CRITIC/ENTROPY + WINNER OUTPUTS
================================================================================

PIPELINE
--------
  Phase 1 - Sweep (~24-30h)
    For each of 1,120 configs (2 enc x 8 PCs x 5 gammas x 7 K_pre x 2 merge states):
      - Run spectral clustering + (optionally) hierarchical merge
      - Compute all clustering quality metrics
      - Cache the labels in a dict keyed on the config tuple

  Phase 2 - CRITIC + Entropy selection (~1 minute)
    On the post-merge eligible subset (30 <= K <= 150, no singletons,
    no clusters < 20):
      - Min-max normalise the four decision metrics (population homogeneity,
        continental homogeneity, NMI vs population, silhouette)
      - Compute CRITIC weights (sigma * (1 - corr))
      - Compute Shannon entropy weights (1 - H_j)
      - Compute composite scores under each scheme
      - Rank, take consensus by mean rank
    Append critic_score, entropy_score, critic_rank, entropy_rank, mean_rank
    columns to all_results.csv (NaN for ineligible rows).

  Phase 3 - Winner regeneration (~10 seconds)
    Look up the cached labels for the winner. Write:
      - cluster_assignments.csv
      - cluster_summary.csv
      - cluster_summary_detailed.txt  (long-form per-cluster breakdown)
      - best_configuration.json

USAGE
-----
  python pipeline_sweep_select_regenerate.py \
      --traw   /path/to/pca_ready.traw \
      --meta   /path/to/metadata.superpop_with_sex.txt \
      --outdir /path/to/output_dir

OUTPUTS
-------
  <outdir>/tables/all_results.csv            (1,120 rows + CRITIC/Entropy columns)
  <outdir>/tables/weights.csv                (4 rows: CRITIC + Entropy weights per metric)
  <outdir>/tables/consensus_ranking.csv      (eligible configs sorted by mean rank)
  <outdir>/tables/run_log.txt                (timestamped log of the whole run)
  <outdir>/winner/cluster_assignments.csv
  <outdir>/winner/cluster_summary.csv
  <outdir>/winner/cluster_summary_detailed.txt
  <outdir>/winner/best_configuration.json

================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

ENCODING_METHODS = ["simple", "twobit"]
PC_VALUES = [5, 7, 10, 15, 20, 30, 50, 100]
GAMMA_VALUES = ["auto", 0.01, 0.1, 1.0, 10.0]
K_VALUES_PRE_MERGE = [50, 75, 100, 125, 150, 175, 200]

MIN_CLUSTER_SIZE = 20
RANDOM_STATE = 42

# Eligibility for CRITIC + Entropy selection
ELIG_K_MIN = 30
ELIG_K_MAX = 150

# The four CRITIC + Entropy decision metrics (all higher-is-better)
DECISION_METRICS = [
    "population_homogeneity_weighted",
    "continental_homogeneity_weighted",
    "nmi_population",
    "silhouette",
]

EPS = 1e-12

# Metadata column auto-detection
SAMPLE_COL = "IID"
POP_COL_CANDIDATES = ["population_code", "population", "POP"]
SUPERPOP_COL_CANDIDATES = [
    "super_population", "superpopulation", "superpop", "SUPERPOP"
]


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(log_path, "a")

    def __call__(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_traw(traw_path: Path) -> tuple[np.ndarray, list[str]]:
    print(f"Loading .traw: {traw_path}")
    df = pd.read_csv(traw_path, sep="\t", low_memory=False)
    sample_cols = list(df.columns[6:])
    print(f"  variants: {len(df):,}  samples: {len(sample_cols):,}")
    sample_ids = [str(c).split("_", 1)[1] if "_" in str(c) else str(c)
                  for c in sample_cols]
    X = df[sample_cols].to_numpy(dtype=np.float32).T
    del df
    gc.collect()
    print(f"  matrix shape: {X.shape}")
    return X, sample_ids


def load_metadata(meta_path: Path, sample_ids: list[str]) -> pd.DataFrame:
    print(f"Loading metadata: {meta_path}")
    meta = pd.read_csv(meta_path, sep="\t", dtype=str)

    pop_col = next((c for c in POP_COL_CANDIDATES if c in meta.columns), None)
    sup_col = next((c for c in SUPERPOP_COL_CANDIDATES if c in meta.columns), None)
    if pop_col is None or sup_col is None:
        raise ValueError(
            f"Could not find population/superpopulation columns. "
            f"Got: {list(meta.columns)}"
        )

    id_col = SAMPLE_COL if SAMPLE_COL in meta.columns else meta.columns[0]
    print(f"  using ID='{id_col}' POP='{pop_col}' SUP='{sup_col}'")

    meta = meta.rename(
        columns={id_col: "sample_id", pop_col: "population",
                 sup_col: "superpopulation"}
    )[["sample_id", "population", "superpopulation"]]

    aligned = pd.DataFrame({"sample_id": sample_ids}).merge(
        meta, on="sample_id", how="left"
    )
    n_missing = int(aligned["population"].isna().sum())
    if n_missing > 0:
        print(f"  WARNING: {n_missing} samples missing metadata; labelling 'UNK'")
        aligned["population"] = aligned["population"].fillna("UNK")
        aligned["superpopulation"] = aligned["superpopulation"].fillna("UNK")
    return aligned


# =============================================================================
# ENCODING
# =============================================================================

def encode_simple(genotypes: np.ndarray) -> np.ndarray:
    X = genotypes.astype(np.float32, copy=True)
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X = np.nan_to_num(X, nan=0.0)
    return X


def encode_twobit(genotypes: np.ndarray) -> np.ndarray:
    G = encode_simple(genotypes)
    n_samples, n_variants = G.shape
    X = np.zeros((n_samples, n_variants * 2), dtype=np.float32)
    X[:, 0::2] = (G >= 1).astype(np.float32)
    X[:, 1::2] = (G == 2).astype(np.float32)
    return X


# =============================================================================
# PCA WITH DETERMINISTIC SIGN CONVENTION
# =============================================================================

def standardise_pca_l2(X: np.ndarray, n_pcs: int) -> np.ndarray:
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_std)
    # Deterministic sign convention: largest |loading| entry forced positive.
    for j in range(X_pca.shape[1]):
        loadings = pca.components_[j]
        max_abs_idx = int(np.argmax(np.abs(loadings)))
        if loadings[max_abs_idx] < 0:
            X_pca[:, j] *= -1.0
    X_norm = normalize(X_pca, norm="l2", axis=1)
    return X_norm.astype(np.float32)


# =============================================================================
# GAMMA + SPECTRAL
# =============================================================================

def compute_gamma(pcs: np.ndarray, method) -> float:
    if method == "auto":
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(len(pcs), min(1000, len(pcs)), replace=False)
        dists = cdist(pcs[sample_idx], pcs[sample_idx])
        median_dist = float(np.median(dists[np.triu_indices_from(dists, k=1)]))
        return 1.0 / (2.0 * median_dist ** 2)
    return float(method)


def run_spectral_clustering(pcs: np.ndarray, k: int, gamma: float) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=k,
        affinity="rbf",
        gamma=gamma,
        random_state=RANDOM_STATE,
        n_jobs=1,
        assign_labels="kmeans",
    )
    return sc.fit_predict(pcs)


# =============================================================================
# HIERARCHICAL MERGE (SUPERPOP-CONSTRAINED)
# =============================================================================

def hierarchical_merge_constrained(
    labels: np.ndarray,
    pcs: np.ndarray,
    superpops: np.ndarray,
    min_size: int = MIN_CLUSTER_SIZE,
    max_iterations: int = 500,
) -> np.ndarray:
    labels_new = labels.copy()
    for _ in range(max_iterations):
        sizes = Counter(labels_new)
        small = [c for c, s in sizes.items() if s < min_size]
        if not small:
            break

        info = {}
        for c in sizes.keys():
            mask = labels_new == c
            info[c] = {
                "centroid": pcs[mask].mean(axis=0),
                "size": sizes[c],
                "superpop": Counter(superpops[mask]).most_common(1)[0][0],
            }

        smallest = min(small, key=lambda c: sizes[c])
        small_cent = info[smallest]["centroid"]
        small_sup = info[smallest]["superpop"]

        cands = [(c, v) for c, v in info.items()
                 if c != smallest and v["size"] >= min_size]
        if not cands:
            cands = [(c, v) for c, v in info.items() if c != smallest]
        if not cands:
            break

        dists = []
        for c, v in cands:
            d = np.linalg.norm(small_cent - v["centroid"])
            adj = d if v["superpop"] == small_sup else d + 1000.0
            dists.append((c, adj))
        dists.sort(key=lambda x: x[1])
        target = dists[0][0]

        labels_new[labels_new == smallest] = target

        unique = np.unique(labels_new)
        relabel = {old: new for new, old in enumerate(unique)}
        labels_new = np.array([relabel[l] for l in labels_new])

    return labels_new


# =============================================================================
# METRICS
# =============================================================================

def compute_all_metrics(
    pcs: np.ndarray,
    labels: np.ndarray,
    populations: np.ndarray,
    superpops: np.ndarray,
) -> dict:
    metrics: dict = {}
    n_clusters = int(len(np.unique(labels)))
    metrics["n_clusters"] = n_clusters

    sizes = [int((labels == c).sum()) for c in np.unique(labels)]
    metrics["min_cluster_size"] = int(min(sizes))
    metrics["max_cluster_size"] = int(max(sizes))
    metrics["mean_cluster_size"] = float(np.mean(sizes))
    metrics["median_cluster_size"] = float(np.median(sizes))
    metrics["std_cluster_size"] = float(np.std(sizes))
    metrics["n_singletons"] = int(sum(1 for s in sizes if s == 1))
    metrics["n_below_20"] = int(sum(1 for s in sizes if s < 20))
    metrics["n_valid_clusters"] = int(sum(1 for s in sizes if s >= 20))

    if n_clusters > 1 and n_clusters < len(pcs):
        try:
            metrics["silhouette"] = float(silhouette_score(pcs, labels))
        except Exception:
            metrics["silhouette"] = -1.0
        try:
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(pcs, labels)
            )
        except Exception:
            metrics["calinski_harabasz"] = 0.0
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(pcs, labels))
        except Exception:
            metrics["davies_bouldin"] = float("inf")
    else:
        metrics["silhouette"] = -1.0
        metrics["calinski_harabasz"] = 0.0
        metrics["davies_bouldin"] = float("inf")

    pop_homog, pop_homog_w_num = [], 0.0
    for c in np.unique(labels):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            continue
        cnt = Counter(populations[mask])
        pct = max(cnt.values()) / n * 100.0
        pop_homog.append(pct)
        pop_homog_w_num += pct * n
    metrics["population_homogeneity_mean"] = (
        float(np.mean(pop_homog)) if pop_homog else 0.0
    )
    metrics["population_homogeneity_weighted"] = (
        float(pop_homog_w_num / len(labels)) if len(labels) else 0.0
    )
    metrics["population_homogeneity_median"] = (
        float(np.median(pop_homog)) if pop_homog else 0.0
    )

    cont_homog, cont_homog_w_num = [], 0.0
    for c in np.unique(labels):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            continue
        cnt = Counter(superpops[mask])
        pct = max(cnt.values()) / n * 100.0
        cont_homog.append(pct)
        cont_homog_w_num += pct * n
    metrics["continental_homogeneity_mean"] = (
        float(np.mean(cont_homog)) if cont_homog else 0.0
    )
    metrics["continental_homogeneity_weighted"] = (
        float(cont_homog_w_num / len(labels)) if len(labels) else 0.0
    )

    metrics["nmi_population"] = float(
        normalized_mutual_info_score(populations, labels)
    )
    metrics["ari_population"] = float(
        adjusted_rand_score(populations, labels)
    )
    metrics["nmi_superpop"] = float(
        normalized_mutual_info_score(superpops, labels)
    )
    metrics["ari_superpop"] = float(adjusted_rand_score(superpops, labels))

    try:
        contingency = pd.crosstab(labels, populations)
        chi2, pval, _, _ = stats.chi2_contingency(contingency)
        metrics["chi2_population"] = float(chi2)
        metrics["chi2_pval"] = float(pval)
        n_total = int(contingency.values.sum())
        min_dim = min(contingency.shape) - 1
        metrics["cramers_v"] = (
            float(np.sqrt(chi2 / (n_total * min_dim))) if min_dim > 0 else 0.0
        )
    except Exception:
        metrics["chi2_population"] = 0.0
        metrics["chi2_pval"] = 1.0
        metrics["cramers_v"] = 0.0

    return metrics


def compute_bio_score(metrics: dict, penalize_small: bool = True) -> float:
    """Legacy bio_score formula (reported alongside, NOT used for selection)."""
    score = 0.0
    score += 0.30 * (metrics.get("continental_homogeneity_weighted", 0) / 100.0)
    score += 0.25 * (metrics.get("population_homogeneity_weighted", 0) / 100.0)
    score += 0.20 * metrics.get("nmi_population", 0)
    sil = metrics.get("silhouette", -1)
    sil_norm = max(0.0, (sil + 1.0) / 2.0)
    score += 0.15 * sil_norm
    score += 0.10 * metrics.get("cramers_v", 0)
    if penalize_small:
        penalty = (
            0.01 * metrics.get("n_below_20", 0)
            + 0.02 * metrics.get("n_singletons", 0)
        )
        score = max(0.0, score - penalty)
    return score


# =============================================================================
# CRITIC + ENTROPY (run on the eligible subset of the sweep)
# =============================================================================

def normalise_metrics(eligible: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    norm = pd.DataFrame(index=eligible.index)
    for m in metrics:
        v = eligible[m].astype(float).values
        v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))
        rng = v_max - v_min
        norm[m] = 0.5 if rng < EPS else (v - v_min) / rng
    return norm


def critic_weights(norm_df: pd.DataFrame) -> dict[str, float]:
    metrics = list(norm_df.columns)
    sigma = norm_df.std(ddof=1).values
    corr = norm_df.corr(method="pearson").values
    conflict = (1.0 - corr).sum(axis=1)
    information = sigma * conflict
    if information.sum() < EPS:
        w = np.full(len(metrics), 1.0 / len(metrics))
    else:
        w = information / information.sum()
    return dict(zip(metrics, w))


def entropy_weights(norm_df: pd.DataFrame) -> dict[str, float]:
    metrics = list(norm_df.columns)
    n = len(norm_df)
    k = 1.0 / np.log(n)
    weights = []
    for m in metrics:
        x = norm_df[m].values
        col_sum = x.sum()
        if col_sum < EPS:
            weights.append(0.0)
            continue
        p = x / col_sum
        p_safe = np.where(p > 0, p, EPS)
        H = -k * np.sum(p * np.log(p_safe))
        d = 1.0 - H
        weights.append(max(d, 0.0))
    weights = np.array(weights)
    if weights.sum() < EPS:
        w = np.full(len(metrics), 1.0 / len(metrics))
    else:
        w = weights / weights.sum()
    return dict(zip(metrics, w))


def run_critic_entropy_selection(
    df: pd.DataFrame,
    log,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], pd.DataFrame, pd.Series]:
    """Compute CRITIC + Entropy ranking on the eligible subset.

    Returns:
      df_with_ranks    : original df with critic_score, entropy_score,
                         critic_rank, entropy_rank, mean_rank columns added
                         (NaN for ineligible rows)
      w_critic         : CRITIC weights
      w_entropy        : Entropy weights
      consensus        : eligible configs sorted by mean rank
      winner_row       : the winning row (rank 1 by mean rank)
    """
    n_total = len(df)
    eligible_mask = (
        (df["merged"] == True)  # noqa: E712
        & df["n_clusters"].between(ELIG_K_MIN, ELIG_K_MAX)
        & (df["n_singletons"] == 0)
        & (df["n_below_20"] == 0)
    )
    eligible_idx = df.index[eligible_mask]
    n_eligible = int(eligible_mask.sum())
    log(
        f"Eligible after filter (merged & {ELIG_K_MIN}<=K<={ELIG_K_MAX} & "
        f"no singletons & no clusters<20): {n_eligible} / {n_total}"
    )
    if n_eligible < 2:
        raise RuntimeError(
            "Fewer than 2 eligible configurations - cannot compute weights."
        )

    eligible = df.loc[eligible_idx].copy()

    # Normalise + weights
    norm = normalise_metrics(eligible, DECISION_METRICS)
    w_critic = critic_weights(norm)
    w_entropy = entropy_weights(norm)

    log("CRITIC weights:")
    for m, w in w_critic.items():
        log(f"  {m:40s}  {w:.4f}")
    log("Entropy weights:")
    for m, w in w_entropy.items():
        log(f"  {m:40s}  {w:.4f}")

    # Composite scores (only meaningful for eligible rows)
    crit_score = np.zeros(len(eligible))
    ent_score = np.zeros(len(eligible))
    for m in DECISION_METRICS:
        crit_score += w_critic[m] * norm[m].values
        ent_score += w_entropy[m] * norm[m].values

    eligible["critic_score"] = crit_score
    eligible["entropy_score"] = ent_score
    eligible["critic_rank"] = (
        eligible["critic_score"].rank(ascending=False, method="min").astype(int)
    )
    eligible["entropy_rank"] = (
        eligible["entropy_score"].rank(ascending=False, method="min").astype(int)
    )
    eligible["mean_rank"] = (
        eligible["critic_rank"] + eligible["entropy_rank"]
    ) / 2.0

    # Add columns to the full df (NaN for ineligible)
    df_out = df.copy()
    for col in [
        "critic_score", "entropy_score",
        "critic_rank", "entropy_rank", "mean_rank",
    ]:
        df_out[col] = np.nan
        df_out.loc[eligible_idx, col] = eligible[col].values

    consensus = eligible.sort_values(
        ["mean_rank", "critic_score"], ascending=[True, False]
    ).reset_index(drop=False).rename(columns={"index": "all_results_index"})

    winner_row = consensus.iloc[0]

    log("Top 5 consensus (mean rank across CRITIC and Entropy):")
    for i, r in consensus.head(5).iterrows():
        log(
            f"  rank {i+1}: {r['encoding']} | {int(r['n_pcs'])} PCs | "
            f"gamma={r['gamma_method']} ({float(r['gamma']):.4f}) | "
            f"K_pre={int(r['k_pre_merge'])} -> K={int(r['n_clusters'])} | "
            f"critic={r['critic_score']:.4f} entropy={r['entropy_score']:.4f}"
        )

    return df_out, w_critic, w_entropy, consensus, winner_row


# =============================================================================
# WINNER OUTPUT FILES
# =============================================================================

def write_cluster_assignments(
    meta: pd.DataFrame,
    labels_original: np.ndarray,
    labels_merged: np.ndarray,
    outdir: Path,
) -> pd.DataFrame:
    df = meta.copy()
    df["cluster_original"] = labels_original
    df["cluster_merged"] = labels_merged
    out = outdir / "cluster_assignments.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out}  ({len(df)} rows)")
    return df


def write_cluster_summary(
    assignments: pd.DataFrame, outdir: Path
) -> pd.DataFrame:
    rows = []
    for c, sub in assignments.groupby("cluster_merged"):
        pop_counts = Counter(sub["population"])
        sup_counts = Counter(sub["superpopulation"])
        dom_pop, dom_pop_n = pop_counts.most_common(1)[0]
        dom_sup, dom_sup_n = sup_counts.most_common(1)[0]
        n = len(sub)
        rows.append({
            "cluster": c,
            "n_samples": n,
            "n_populations": len(pop_counts),
            "dominant_population": dom_pop,
            "dominant_pop_count": dom_pop_n,
            "dominant_pop_pct": 100.0 * dom_pop_n / n,
            "dominant_superpop": dom_sup,
            "dominant_superpop_count": dom_sup_n,
            "dominant_superpop_pct": 100.0 * dom_sup_n / n,
            "all_populations": dict(pop_counts),
            "all_superpops": dict(sup_counts),
        })
    df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    out = outdir / "cluster_summary.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out}  ({len(df)} clusters)")
    return df


def write_detailed_summary(
    assignments: pd.DataFrame,
    outdir: Path,
    encoding: str,
    n_pcs: int,
) -> None:
    sizes = assignments.groupby("cluster_merged").size()
    n_clusters = sizes.shape[0]

    lines = []
    p = lines.append
    p("=" * 100)
    p("CLUSTER SUMMARY - BEST CONFIGURATION")
    p(f"Encoding: {encoding}, PCs: {n_pcs}, K: {n_clusters}")
    p("=" * 100)
    p(f"Total samples: {len(assignments)}")
    p(f"Total clusters: {n_clusters}")
    p(f"Min cluster size: {int(sizes.min())}")
    p(f"Max cluster size: {int(sizes.max())}")
    p(f"Mean cluster size: {sizes.mean():.1f}")

    for cluster_id in sorted(assignments["cluster_merged"].unique()):
        sub = assignments[assignments["cluster_merged"] == cluster_id]
        n = len(sub)
        sup_counts = Counter(sub["superpopulation"])
        pop_counts = Counter(sub["population"])

        p("-" * 80)
        p(f"CLUSTER {int(cluster_id)}: {n} samples")
        p("-" * 80)

        p("  SUPERPOPULATION BREAKDOWN:")
        for sup, cnt in sup_counts.most_common():
            p(f"    {sup}: {cnt} ({100.0 * cnt / n:.1f}%)")

        p("  POPULATION BREAKDOWN:")
        for pop, cnt in pop_counts.most_common():
            p(f"    {pop}: {cnt} ({100.0 * cnt / n:.1f}%)")

    out = outdir / "cluster_summary_detailed.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def write_best_configuration_json(
    winner_row: pd.Series,
    n_clusters: int,
    outdir: Path,
) -> None:
    record = {
        "encoding": str(winner_row["encoding"]),
        "n_pcs": int(winner_row["n_pcs"]),
        "gamma_method": str(winner_row["gamma_method"]),
        "gamma": float(winner_row["gamma"]),
        "k_pre_merge": int(winner_row["k_pre_merge"]),
        "k_post_merge": int(n_clusters),
        "min_cluster_size_param": MIN_CLUSTER_SIZE,
        "selection_method": "CRITIC + Entropy consensus (mean rank)",
        "critic_score": float(winner_row["critic_score"]),
        "entropy_score": float(winner_row["entropy_score"]),
        "critic_rank": int(winner_row["critic_rank"]),
        "entropy_rank": int(winner_row["entropy_rank"]),
        "mean_rank": float(winner_row["mean_rank"]),
        "random_state": RANDOM_STATE,
    }
    out = outdir / "best_configuration.json"
    with open(out, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"  wrote {out}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traw", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.outdir / "tables"
    winner_dir = args.outdir / "winner"
    tables_dir.mkdir(parents=True, exist_ok=True)
    winner_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(tables_dir / "run_log.txt")
    log("=" * 80)
    log("END-TO-END PIPELINE: SWEEP + CRITIC/ENTROPY + WINNER OUTPUTS")
    log("=" * 80)
    log(f"input traw : {args.traw}")
    log(f"input meta : {args.meta}")
    log(f"output dir : {args.outdir}")
    log(f"random_state: {RANDOM_STATE}")
    log("=" * 80)

    np.random.seed(RANDOM_STATE)

    # Pin BLAS at the env-var level (must be done in shell, but reinforce here
    # in case the user forgot, using threadpool_limits below as the actual lock)
    for var in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(var, "1")

    log("Pinning BLAS to 1 thread for deterministic numerics")
    with threadpool_limits(limits=1):
        # =====================================================================
        # PHASE 1 - SWEEP
        # =====================================================================
        log("")
        log("=" * 80)
        log("PHASE 1 - PARAMETER SWEEP")
        log("=" * 80)

        log("Loading inputs...")
        G, sample_ids = load_traw(args.traw)
        meta = load_metadata(args.meta, sample_ids)
        populations = meta["population"].to_numpy()
        superpops = meta["superpopulation"].to_numpy()

        log("Pre-encoding (simple + twobit)...")
        encodings = {
            "simple": encode_simple(G),
            "twobit": encode_twobit(G),
        }
        del G
        gc.collect()
        for k, v in encodings.items():
            log(f"  {k}: {v.shape}, ~{v.nbytes/1e9:.2f} GB")

        combos = [
            (enc, npc, gv, kv)
            for enc in ENCODING_METHODS
            for npc in PC_VALUES
            for gv in GAMMA_VALUES
            for kv in K_VALUES_PRE_MERGE
        ]
        total_configs = len(combos)
        log(f"Total parameter combinations: {total_configs}  "
            f"(-> {total_configs * 2} rows in CSV)")

        rows: list[dict] = []
        # Cache labels for every (config, merged_flag) so we can grab the
        # winner's labels later without re-clustering.
        label_cache: dict[tuple, np.ndarray] = {}

        # Per-encoding PCA cache: keyed on (encoding, n_pcs)
        pca_cache: dict[tuple[str, int], np.ndarray] = {}

        t_start = time.time()

        for combo_idx, (enc, npc, gv, kv) in enumerate(combos, 1):
            cache_key = (enc, npc)
            if cache_key not in pca_cache:
                log(f"PCA cache build: {enc} encoding, {npc} PCs")
                pca_cache[cache_key] = standardise_pca_l2(encodings[enc], npc)
                # Drop unused encodings' PCA caches to limit memory
                for ck in list(pca_cache.keys()):
                    if ck[0] != enc:
                        del pca_cache[ck]
                gc.collect()

            pcs_norm = pca_cache[cache_key]

            try:
                gamma = compute_gamma(pcs_norm, gv)

                labels_pre = run_spectral_clustering(pcs_norm, kv, gamma)
                metrics_pre = compute_all_metrics(
                    pcs_norm, labels_pre, populations, superpops
                )
                bio_pre = compute_bio_score(metrics_pre)

                labels_post = hierarchical_merge_constrained(
                    labels_pre, pcs_norm, superpops, MIN_CLUSTER_SIZE
                )
                metrics_post = compute_all_metrics(
                    pcs_norm, labels_post, populations, superpops
                )
                bio_post = compute_bio_score(metrics_post)

                # Cache labels by (encoding, n_pcs, gamma_method, k_pre, merged)
                key_pre = (enc, npc, str(gv), kv, False)
                key_post = (enc, npc, str(gv), kv, True)
                label_cache[key_pre] = labels_pre.copy()
                label_cache[key_post] = labels_post.copy()

                for merged_flag, M, score in (
                    (False, metrics_pre, bio_pre),
                    (True, metrics_post, bio_post),
                ):
                    row = dict(M)
                    row.update({
                        "bio_score": score,
                        "encoding": enc,
                        "n_pcs": npc,
                        "gamma_method": str(gv),
                        "gamma": gamma,
                        "k_pre_merge": kv,
                        "merged": merged_flag,
                    })
                    rows.append(row)

                elapsed = time.time() - t_start
                avg = elapsed / combo_idx
                remaining = (total_configs - combo_idx) * avg
                log(
                    f"[{combo_idx:4d}/{total_configs}] "
                    f"{enc}/PC{npc}/g={gv}/Kpre={kv}: "
                    f"K_pre={metrics_pre['n_clusters']} -> "
                    f"K_post={metrics_post['n_clusters']}, "
                    f"bio_post={bio_post:.4f} (elapsed {elapsed/60:.1f}m, "
                    f"~{remaining/60:.1f}m remaining)"
                )
            except Exception as e:
                log(f"  WARN: failed {enc}/PC{npc}/g={gv}/Kpre={kv}: {e}")
                continue

            # Periodic checkpoint of the CSV (without ranking columns yet)
            if combo_idx % 20 == 0:
                pd.DataFrame(rows).to_csv(
                    tables_dir / "all_results.csv", index=False
                )

        # Free encoding matrices and PCA caches now that we have the labels
        del encodings
        del pca_cache
        gc.collect()

        df = pd.DataFrame(rows)
        log(f"Sweep complete: {len(df)} rows")

        # =====================================================================
        # PHASE 2 - CRITIC + ENTROPY SELECTION
        # =====================================================================
        log("")
        log("=" * 80)
        log("PHASE 2 - CRITIC + ENTROPY SELECTION")
        log("=" * 80)

        df_with_ranks, w_critic, w_entropy, consensus, winner_row = (
            run_critic_entropy_selection(df, log)
        )

        # Write all_results.csv WITH the ranking columns
        all_results_path = tables_dir / "all_results.csv"
        df_with_ranks.to_csv(all_results_path, index=False)
        log(f"Wrote {all_results_path} with CRITIC/Entropy columns")

        # Write supporting files
        weights_df = pd.DataFrame({
            "metric": DECISION_METRICS,
            "critic_weight": [w_critic[m] for m in DECISION_METRICS],
            "entropy_weight": [w_entropy[m] for m in DECISION_METRICS],
        })
        weights_df.to_csv(tables_dir / "weights.csv", index=False)
        log(f"Wrote {tables_dir / 'weights.csv'}")

        consensus.to_csv(tables_dir / "consensus_ranking.csv", index=False)
        log(f"Wrote {tables_dir / 'consensus_ranking.csv'}")

        log("=" * 80)
        log("WINNER (consensus by mean rank across CRITIC and Entropy):")
        log(f"  encoding      : {winner_row['encoding']}")
        log(f"  n_pcs         : {int(winner_row['n_pcs'])}")
        log(f"  gamma_method  : {winner_row['gamma_method']}")
        log(f"  gamma         : {float(winner_row['gamma']):.4f}")
        log(f"  k_pre_merge   : {int(winner_row['k_pre_merge'])}")
        log(f"  n_clusters    : {int(winner_row['n_clusters'])}")
        log(f"  critic_score  : {winner_row['critic_score']:.4f} "
            f"(rank {int(winner_row['critic_rank'])})")
        log(f"  entropy_score : {winner_row['entropy_score']:.4f} "
            f"(rank {int(winner_row['entropy_rank'])})")
        log(f"  mean_rank     : {winner_row['mean_rank']:.2f}")
        log("=" * 80)

        # =====================================================================
        # PHASE 3 - WINNER OUTPUT FILES (from cached labels - no re-clustering)
        # =====================================================================
        log("")
        log("=" * 80)
        log("PHASE 3 - WINNER OUTPUT FILES")
        log("=" * 80)

        winner_key_post = (
            str(winner_row["encoding"]),
            int(winner_row["n_pcs"]),
            str(winner_row["gamma_method"]),
            int(winner_row["k_pre_merge"]),
            True,  # merged
        )
        winner_key_pre = (
            winner_key_post[0], winner_key_post[1], winner_key_post[2],
            winner_key_post[3], False,
        )
        if winner_key_post not in label_cache:
            raise RuntimeError(
                f"Winner labels not cached: {winner_key_post}. "
                "This should be impossible if Phase 1 ran cleanly."
            )

        labels_pre = label_cache[winner_key_pre]
        labels_post = label_cache[winner_key_post]

        # Sanity check: K from labels must match K in CSV
        k_from_labels = len(set(labels_post))
        k_from_csv = int(winner_row["n_clusters"])
        if k_from_labels != k_from_csv:
            raise RuntimeError(
                f"K mismatch between cached labels ({k_from_labels}) "
                f"and CSV row ({k_from_csv}). This is a bug."
            )
        log(f"K from cached labels matches K from CSV: {k_from_labels}  ✓")

        # Write the four winner files
        log("Writing winner output files...")
        assignments = write_cluster_assignments(
            meta, labels_pre, labels_post, winner_dir
        )
        write_cluster_summary(assignments, winner_dir)
        write_detailed_summary(
            assignments,
            winner_dir,
            encoding=str(winner_row["encoding"]),
            n_pcs=int(winner_row["n_pcs"]),
        )
        write_best_configuration_json(
            winner_row, n_clusters=k_from_labels, outdir=winner_dir
        )

        log("")
        log("=" * 80)
        log("PIPELINE COMPLETE")
        log("=" * 80)
        log(f"Sweep CSV     : {all_results_path}")
        log(f"Weights       : {tables_dir / 'weights.csv'}")
        log(f"Consensus     : {tables_dir / 'consensus_ranking.csv'}")
        log(f"Winner files  : {winner_dir}")
        log("=" * 80)

    log.close()


if __name__ == "__main__":
    main()
