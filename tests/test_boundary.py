"""Tests for boundary conditions and bulk formulae."""

import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.boundary import bulk_Cd, bulk_Ch, compute_ustar2


def test_bulk_Cd_typical():
    Cd = bulk_Cd(10.0)
    assert 1e-3 < Cd < 3e-3


def test_bulk_Cd_low_wind():
    Cd = bulk_Cd(0.0)
    assert Cd > 0


def test_bulk_Ch_equals_Cd():
    assert bulk_Ch(10.0) == bulk_Cd(10.0)


def test_ustar2_zero_wind():
    ustar2, Cd, ws = compute_ustar2(0.0, 0.0, 0.0, 0.0, 0.5)
    assert ustar2 >= 0


def test_ustar2_typical():
    ustar2, Cd, ws = compute_ustar2(10.0, 0.0, 0.0, 0.0, 10.0)
    assert ustar2 > 0
    assert Cd == pytest.approx(bulk_Cd(10.0))
