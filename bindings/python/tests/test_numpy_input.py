"""The numpy fast path must agree exactly with list input (same core call)."""

import pytest

np = pytest.importorskip("numpy")
import tsorder


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(42)
    x = [0.0, 0.0]
    for _ in range(1500):
        e = rng.standard_normal()
        x.append(1.2 * x[-1] - 0.6 * x[-2] + e)
    return x[100:]


def test_numpy_array_matches_list(series):
    arr = np.asarray(series)  # contiguous float64: zero-copy path
    assert tsorder.fh_order(arr) == tsorder.fh_order(series)
    assert tsorder.ic_order(arr) == tsorder.ic_order(series)
    assert tsorder.jirak_order(arr) == tsorder.jirak_order(series)
    assert tsorder.qn(arr) == tsorder.qn(series)


def test_non_contiguous_and_other_dtypes(series):
    arr = np.asarray(series)
    rev = arr[::-1][::-1]  # non-contiguous view with the original order
    assert not rev.flags["C_CONTIGUOUS"] or rev.base is not None
    assert tsorder.fh_order(rev) == tsorder.fh_order(series)
    f32 = arr.astype(np.float32)
    r = tsorder.fh_order(f32)  # element-wise fallback
    assert r["order"] == tsorder.fh_order(series)["order"]
    ints = (10 * arr).astype(np.int64)  # integer dtype uses the fallback too
    assert tsorder.fh_order(ints)["order"] == tsorder.fh_order((10 * arr))["order"]
