"""§11.4 Leng et al. (IMAV 2024) comparison tests.

Data-dependent: skipped until the original parameter table and suspended-test
time series are supplied (brief §12; action plan §6 open item 1). These skips
must remain until real data arrive — no synthetic stand-in may be labelled
validation (brief §14.5).
"""

import pytest

pytestmark = pytest.mark.skip(reason="awaiting original data — Leng et al. IMAV 2024 (M4)")


def test_leng2024_suspended_comparison():
    raise NotImplementedError


def test_suspended_abdomen_comparison():
    raise NotImplementedError
