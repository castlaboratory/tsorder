# tsorder (Python) — planned

PyO3 + maturin wrapper over `tsorder-core`. Target API:

```python
import tsorder
tsorder.fh_order(x, statistic="pacf", threshold="bic", c=1.5, memory=None)   # -> FhResult dataclass
tsorder.ic_order(x, criterion="bic", max_order=None)
tsorder.ensemble_order(x, statistics=["pacf", "long_ar_ols"], rule="majority")
tsorder.order_select(x, method="pacf")
```

Accepts lists, numpy arrays and pandas Series. Wheels via maturin-action.
