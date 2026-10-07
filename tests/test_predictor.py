from starvoice.predictor import PhaseModel


def test_phase_risk_boundary():
    m = PhaseModel(period_s=15.0, boundary_offset_s=100.0, risk_half_width_s=0.5, score_threshold_ms=50)
    assert m.risk(int(100.0e9)) == 1.0
    assert m.risk(int(107.5e9)) == 0.0
    assert m.risk(int(114.9e9)) > 0.0
