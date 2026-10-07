import pytest

from starvoice.dnsmos import _polyfit


def test_dnsmos_official_nonpersonalized_polyfit():
    sig, bak, ovr = _polyfit(3.0, 3.0, 3.0)
    assert sig == pytest.approx(2.91200747)
    assert bak == pytest.approx(3.24640004)
    assert ovr == pytest.approx(2.78345392)
