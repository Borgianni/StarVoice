from starvoice.wer import _paired, _score


def test_score_counts_word_errors():
    result = _score("the quick brown fox", "the fast brown")
    assert result["reference_words"] == 4
    assert result["substitutions"] == 1
    assert result["deletions"] == 1
    assert result["insertions"] == 0
    assert result["errors"] == 2
    assert result["wer"] == 0.5


def test_paired_lower_wer_is_better():
    rows = [
        {
            "trace_name": "trace",
            "utterance_id": "utt",
            "policy": "predictive-fec",
            "actual_wer": 0.1,
            "actual_errors": 1,
        },
        {
            "trace_name": "trace",
            "utterance_id": "utt",
            "policy": "random-fec",
            "actual_wer": 0.2,
            "actual_errors": 2,
        },
    ]
    result = _paired(rows, "actual_wer", "predictive-fec", "random-fec")
    assert result["pairs"] == 1
    assert result["missing_pairs"] == 0
    assert result["left_better"] == 1
    assert result["left_worse"] == 0
    assert result["total_error_delta"] == -1
