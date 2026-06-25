# ancestry-clustering-pipeline

Reproducible end-to-end pipeline for spectral-clustering-based ancestry reference panel construction, with objective CRITIC/entropy model selection over 1,120 configurations.

## Overview


1. **Sweeps** 1,120 configurations (2 encodings × 8 PC counts × 5 gamma values × 7 pre-merge K values), running spectral clustering plus optional superpopulation-constrained hierarchical merging on each, computing clustering-quality metrics, and caching every label array in memory.
2. **Selects** a winning configuration objectively using CRITIC and Shannon-entropy weighting over four decision metrics (population homogeneity, continental homogeneity, NMI, silhouette), ranking the eligible subset by consensus mean rank.
3. **Regenerates** the winner's per-cluster output files.

The central design guarantee is reproducibility: the cluster assignments written for the winner come from the **same in-memory labels** used to compute the metrics, so K can never drift between the sweep CSV and the per-cluster output files.

## Reproducibility

- BLAS pinned to 1 thread via `threadpoolctl` and environment variables.
- PCA components forced to a deterministic sign convention (largest absolute loading forced positive).
- `random_state=42` throughout, including the gamma `auto` heuristic.
- Superpopulation-constrained hierarchical merge (Euclidean centroid distance with a cross-superpopulation penalty).

## Installation

Python 3.9+ recommended.

```bash
git clone https://github.com/Taghrid-M/ancestry-clustering-pipeline.git
cd ancestry-clustering-pipeline
pip install -r requirements.txt
```

## Usage

```bash
python pipeline_sweep_select_regenerate.py \
    --traw   /path/to/pca_ready.traw \
    --meta   /path/to/metadata.superpop_with_sex.txt \
    --outdir /path/to/output_dir
```

The full sweep runs for approximately 24–30 hours on a single node. BLAS is pinned to one thread for determinism; for best results, also set the thread environment variables in your shell before launching:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### Inputs

**`--traw`** — a PLINK `.traw` genotype dosage matrix (tab-separated). The first 6 columns are variant metadata; remaining columns are samples. Sample IDs are taken from the column headers (the portion after the first underscore).

**`--meta`** — a tab-separated metadata file with one row per sample. It must contain:
- a sample-ID column (`IID`, or the first column),
- a population column (one of `population_code`, `population`, `POP`),
- a superpopulation column (one of `super_population`, `superpopulation`, `superpop`, `SUPERPOP`).

Samples present in the genotype matrix but missing from the metadata are labelled `UNK`.

### Outputs

```
<outdir>/tables/all_results.csv          # 1,120 configs × metrics + CRITIC/entropy columns
<outdir>/tables/weights.csv              # CRITIC and entropy weights per metric
<outdir>/tables/consensus_ranking.csv    # eligible configs sorted by mean rank
<outdir>/tables/run_log.txt              # timestamped run log
<outdir>/winner/cluster_assignments.csv
<outdir>/winner/cluster_summary.csv
<outdir>/winner/cluster_summary_detailed.txt
<outdir>/winner/best_configuration.json
```

## Selection method

Eligibility for selection: post-merge configurations with 30 ≤ K ≤ 150, no singleton clusters, and no clusters smaller than 20 samples. On this subset the four decision metrics are min-max normalised, then weighted two ways:

- **CRITIC** — weights proportional to each metric's standard deviation times its conflict (one minus correlation) with the others.
- **Shannon entropy** — weights proportional to one minus the normalised entropy of each metric.

Composite scores are computed under each scheme, configurations are ranked under each, and the winner is the configuration with the best consensus (mean) rank.

A legacy `bio_score` is also reported in `all_results.csv` for reference but is **not** used for selection.

