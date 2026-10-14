import json
import threading

import pytest

from starvoice import storage


def test_async_jsonl_preserves_order_and_flushes(tmp_path):
    path = tmp_path / "received.jsonl"
    writer = storage.AsyncJsonlWriter(path)
    for sequence in range(1000):
        writer.write({"sequence": sequence})
    writer.close()
    assert [json.loads(x)["sequence"] for x in path.read_text().splitlines()] == list(
        range(1000)
    )
    assert writer.stats["max_queue_depth"] >= 1
    with pytest.raises(RuntimeError, match="closed"):
        writer.write({"sequence": 1000})


def test_async_jsonl_does_not_block_producer_on_slow_disk(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    observed = []

    class SlowSink:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def write(self, obj):
            started.set()
            if not release.wait(2):
                raise TimeoutError("test sink was not released")
            observed.append(obj["sequence"])

    monkeypatch.setattr(storage, "JsonlWriter", SlowSink)
    writer = storage.AsyncJsonlWriter(tmp_path / "slow.jsonl", max_pending=3)
    try:
        writer.write({"sequence": 0})
        assert started.wait(1), "worker never began simulated disk write"
        writer.write({"sequence": 1})
        writer.write({"sequence": 2})
        writer.write({"sequence": 3})
        with pytest.raises(RuntimeError, match="queue full"):
            writer.write({"sequence": 4})
    finally:
        release.set()
        writer.close()
    assert observed == [0, 1, 2, 3]
