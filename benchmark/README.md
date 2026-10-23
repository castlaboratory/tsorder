# benchmark

Reproducible order-selection benchmark. The runner is `crates/tsorder-bench`
(rayon + TOML designs; seeds are a pure function of (dgp, n, replication), so
every estimator sees the same series and reruns are bit-identical).

```sh
# run one or more designs (writes benchmark/results/*.csv, aggregated losslessly)
cargo run --release -p tsorder-bench -- benchmark/configs/q1.toml

# figures (benchmark/figures/, not versioned; needs R with dplyr/ggplot2/readr/tidyr)
Rscript benchmark/analysis/q1.R
```

First round (2026-10-23, AR/MA): Q1 accuracy/level by family, Q2 ensembles,
Q3 contamination, Q5 CPU cost, Q6 prediction under infinite order, Q7 memory
rule. The ARMA round (Q4, Q8) comes with the McMillan-degree estimators.
`results/*.csv` are the versioned raw outputs each figure is built from.
