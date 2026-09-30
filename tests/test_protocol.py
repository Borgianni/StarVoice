from starvoice.protocol import KIND_PROBE, Packet, decode_packet, encode_packet


def test_roundtrip():
    p = Packet(KIND_PROBE, 3, 42, 123456789, b"hello")
    assert decode_packet(encode_packet(p)) == p
