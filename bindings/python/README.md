# tsorder (Python)

Thin PyO3 shell over the dependency-free Rust core `tsorder-core` (no
statistical logic lives here). Accepts lists, numpy arrays or pandas
Series; returns plain dicts. Bit-identical to the R package by
construction — both replay the same golden fixture in their test suites.

```python
# from the repository checkout (needs Rust >= 1.81):
#   pip install maturin && maturin develop --release -m bindings/python/Cargo.toml
import tsorder

fit = tsorder.fh_order(x)                 # PACF-clip, the default
fit["order"], fit["tau"]

tsorder.ic_order(x, criterion="bic")      # classical criteria
tsorder.svc_order(x)                      # McMillan degree
tsorder.integration_order(x)              # how many differences
tsorder.hannan_rissanen(x)                # ARMA (p, q)
```

See the [project page](https://castlaboratory.github.io/tsorder/) and the
[R package site](https://castlaboratory.github.io/tsorder/r/).
