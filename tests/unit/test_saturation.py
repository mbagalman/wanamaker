
import numpy as np
import pytest
from wanamaker.transforms.saturation import hill_saturation

def test_hill_saturation_basic():
    spend = np.array([0.0, 10.0, 50.0, 100.0, 1000.0])
    ec50 = 50.0
    slope = 1.0

    # f(x) = x^1 / (x^1 + 50^1)
    # 0 -> 0 / 50 = 0
    # 10 -> 10 / 60 = 1/6 ~ 0.1666
    # 50 -> 50 / 100 = 0.5
    # 100 -> 100 / 150 = 2/3 ~ 0.666
    # 1000 -> 1000 / 1050 = 20/21 ~ 0.952

    out = hill_saturation(spend, ec50, slope)

    expected = np.array([0.0, 10/60, 0.5, 100/150, 1000/1050])
    np.testing.assert_allclose(out, expected, rtol=1e-6)

def test_hill_saturation_slope():
    spend = np.array([10.0, 50.0, 100.0])
    ec50 = 50.0
    slope = 2.0

    # f(x) = x^2 / (x^2 + 50^2) = x^2 / (x^2 + 2500)
    # 10 -> 100 / 2600 = 1/26 ~ 0.03846
    # 50 -> 2500 / 5000 = 0.5
    # 100 -> 10000 / 12500 = 0.8

    out = hill_saturation(spend, ec50, slope)
    expected = np.array([1/26, 0.5, 0.8])
    np.testing.assert_allclose(out, expected, rtol=1e-6)

def test_hill_saturation_invalid_params():
    spend = np.array([10.0, 20.0])

    with pytest.raises(ValueError, match="ec50 must be strictly positive"):
        hill_saturation(spend, -1.0, 1.0)

    with pytest.raises(ValueError, match="slope must be strictly positive"):
        hill_saturation(spend, 50.0, -1.0)
