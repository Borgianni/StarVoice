import pytest

from starvoice.opus import OpusConfig, OpusEncoder, OpusDecoder, OpusUnavailable


def test_opus_silence_roundtrip():
    cfg = OpusConfig()
    try:
        enc = OpusEncoder(cfg)
        dec = OpusDecoder(cfg)
    except OpusUnavailable:
        pytest.skip("libopus unavailable")
    try:
        pcm = b"\0" * (cfg.frame_samples * cfg.channels * 2)
        payload = enc.encode(pcm)
        decoded = dec.decode(payload)
        assert len(decoded) == len(pcm)
        assert payload
    finally:
        enc.close()
        dec.close()
