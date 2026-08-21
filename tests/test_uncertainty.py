import numpy as np

from pycnotide import block_bootstrap_indices


def test_block_bootstrap_is_deterministic_and_complete():
    time = np.arange(100, dtype=float) * 3600.0
    first = block_bootstrap_indices(time, n_resamples=10, seed=7)
    second = block_bootstrap_indices(time, n_resamples=10, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (10, 100)
    assert np.all((first >= 0) & (first < 100))
