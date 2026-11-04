"""§11.4 free-flight validation. Skipped until free-flight data exist."""

import pytest

pytestmark = pytest.mark.skip(
    reason="no free-flight dataset exists for these vehicles (2026-11-04)")


def test_free_flight_comparison():
    raise NotImplementedError
