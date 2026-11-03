"""§11.4 free-flight validation. Skipped until free-flight data exist."""

import pytest

pytestmark = pytest.mark.skip(reason="awaiting original data — free-flight dataset (M4+)")


def test_free_flight_comparison():
    raise NotImplementedError
