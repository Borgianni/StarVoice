"""STARVOICE's SV02 UDP CLI must run on a minimal Ubuntu relay install."""

import subprocess
import sys


def test_cli_loads_without_analysis_extras():
    code = """
import builtins

original = builtins.__import__
forbidden = {"numpy", "pandas", "scipy", "pystoi", "soundfile",
             "librosa", "onnxruntime", "faster_whisper", "matplotlib",
             "pyarrow", "torch", "av"}

def restricted_import(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise ImportError("analysis-only module imported by UDP CLI: " + name)
    return original(name, *args, **kwargs)

builtins.__import__ = restricted_import

from starvoice.cli import _parser
parser = _parser()
args = parser.parse_args(
    ["receive-speech", "--bind", "0.0.0.0", "--port", "5005",
     "--output-dir", "/tmp/test-rx"]
)
assert args.cmd == "receive-speech"
assert args.idle_timeout_s == 600.0
print("MINIMAL_CLI_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MINIMAL_CLI_OK" in result.stdout
