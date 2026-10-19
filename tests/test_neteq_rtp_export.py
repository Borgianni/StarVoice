"""SV02 -> neteq_rtpplay standard RTPdump interoperability preflight.

These tests validate the binary file layout and recorded-arrival semantics.
They DO NOT substitute for actually running native NetEQ.
"""
import struct

import pytest

from starvoice import neteq_rtp_export as export


def decode_rtpdump(raw):
    line_end = raw.index(b"\n") + 1
    assert raw[:line_end] == export.DUMP_LINE
    sec, usec, ipv4, port, padding = struct.unpack_from("!IIIHH", raw, line_end)
    assert (sec, usec, ipv4, port, padding) == (0, 0, 0x7F000001, 5004, 0)
    at = line_end + 16
    parsed = []
    while at < len(raw):
        length, original, offset = struct.unpack_from("!HHI", raw, at)
        assert length >= 20
        assert original == length - 8
        at += 8
        packet = raw[at:at + original]
        assert len(packet) == original
        version, pt, seq, timestamp, ssrc = struct.unpack_from("!BBHII", packet)
        assert version == 0x80
        assert pt == 111
        assert ssrc == export.SSRC
        parsed.append((seq, timestamp, offset, packet[12:]))
        at += original
    assert at == len(raw)
    return parsed


def test_rtpdump_preserves_arrival_not_sequence_order(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "packet_has_fec", lambda pkt: False)
    packets = [b"A", b"BC", b"DEF", b"GHIJ"]
    arrivals = {
        0: {"receive_mono_ns": 1_000_000_000, "opus_bytes": 1, "has_lbrr": False},
        2: {"receive_mono_ns": 1_041_999_999, "opus_bytes": 3, "has_lbrr": False},
        1: {"receive_mono_ns": 1_053_500_000, "opus_bytes": 2, "has_lbrr": False},
    }
    dst = tmp_path / "native.rtp"
    record = export.convert_arrivals(arrivals, packets, dst, first_seq=0)
    parsed = decode_rtpdump(dst.read_bytes())
    assert parsed == [
        (0, 0, 0, b"A"),
        (2, 1920, 41, b"DEF"),
        (1, 960, 53, b"BC"),
    ]
    assert record["observed_rtp_packets"] == 3
    assert record["unobserved_rtp_packets"] == 1
    assert not record["original_opus_packet_bytes_retained"]
    assert not record["external_native_neteq_executed"] if "external_native_neteq_executed" in record else True


def test_receiver_wraps_rtp_sequence_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "packet_has_fec", lambda pkt: False)
    n = 65538
    packets = [b"a"] * n
    seq = 65537
    dst = tmp_path / "wrap.rtp"
    export.convert_arrivals({
        seq: {"receive_mono_ns": 3_000_000_000},
    }, packets, dst, first_seq=seq)
    parsed = decode_rtpdump(dst.read_bytes())
    assert parsed == [(seq & 65535, (seq * 960) & 0xFFFFFFFF, 0, b"a")]


def test_rejects_incorrect_lbrr_or_length_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "packet_has_fec", lambda pkt: False)
    for item in ({"has_lbrr": True}, {"opus_bytes": 99}):
        dst = tmp_path / "invalid.rtp"
        with pytest.raises(ValueError, match="mismatch"):
            export.convert_arrivals({
                0: {"receive_mono_ns": 100, **item},
            }, [b"a"], dst)
        assert not dst.exists()


def test_output_will_not_be_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "packet_has_fec", lambda pkt: False)
    dst = tmp_path / "exists.rtp"
    dst.write_bytes(b"original data")
    with pytest.raises(FileExistsError, match="refusing"):
        export.convert_arrivals({0: {"receive_mono_ns": 1}}, [b"a"], dst)
    assert dst.read_bytes() == b"original data"
