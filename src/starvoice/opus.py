from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass

OPUS_OK = 0
OPUS_APPLICATION_VOIP = 2048
OPUS_SET_BITRATE_REQUEST = 4002
OPUS_SET_INBAND_FEC_REQUEST = 4012
OPUS_SET_PACKET_LOSS_PERC_REQUEST = 4014
OPUS_SET_SIGNAL_REQUEST = 4024
OPUS_SIGNAL_VOICE = 3001
OPUS_SET_DRED_DURATION_REQUEST = 4050


class OpusUnavailable(RuntimeError):
    pass


def _lib() -> ctypes.CDLL:
    name = ctypes.util.find_library("opus")
    if not name:
        raise OpusUnavailable("libopus not found; install libopus0/libopus-dev")
    lib = ctypes.CDLL(name)
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int32,
    ]
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_decode.restype = ctypes.c_int
    lib.opus_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.opus_encoder_ctl.restype = ctypes.c_int
    return lib


@dataclass(frozen=True)
class OpusConfig:
    sample_rate: int = 48000
    channels: int = 1
    frame_ms: int = 20
    bitrate: int = 24000

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


class OpusEncoder:
    def __init__(self, config: OpusConfig):
        self.config = config
        self.lib = _lib()
        err = ctypes.c_int()
        self.ptr = self.lib.opus_encoder_create(
            config.sample_rate, config.channels, OPUS_APPLICATION_VOIP, ctypes.byref(err)
        )
        if not self.ptr or err.value != OPUS_OK:
            raise RuntimeError(f"opus_encoder_create failed: {err.value}")
        self._ctl(OPUS_SET_BITRATE_REQUEST, config.bitrate)
        self._ctl(OPUS_SET_SIGNAL_REQUEST, OPUS_SIGNAL_VOICE)

    def _ctl(self, request: int, value: int) -> None:
        rc = self.lib.opus_encoder_ctl(self.ptr, request, ctypes.c_int(value))
        if rc != OPUS_OK:
            raise RuntimeError(f"opus_encoder_ctl({request}) failed: {rc}")

    def set_fec(self, enabled: bool, expected_loss_percent: int = 0) -> None:
        self._ctl(OPUS_SET_INBAND_FEC_REQUEST, int(enabled))
        self._ctl(OPUS_SET_PACKET_LOSS_PERC_REQUEST, expected_loss_percent if enabled else 0)

    def set_dred_duration(self, frames_10ms: int) -> bool:
        rc = self.lib.opus_encoder_ctl(
            self.ptr, OPUS_SET_DRED_DURATION_REQUEST, ctypes.c_int(frames_10ms)
        )
        return rc == OPUS_OK

    def encode(self, pcm: bytes) -> bytes:
        expected = self.config.frame_samples * self.config.channels * 2
        if len(pcm) != expected:
            raise ValueError(f"expected {expected} PCM bytes, got {len(pcm)}")
        samples = (ctypes.c_int16 * (len(pcm) // 2)).from_buffer_copy(pcm)
        out = (ctypes.c_ubyte * 4000)()
        n = self.lib.opus_encode(self.ptr, samples, self.config.frame_samples, out, len(out))
        if n < 0:
            raise RuntimeError(f"opus_encode failed: {n}")
        return bytes(out[:n])

    def close(self) -> None:
        if self.ptr:
            self.lib.opus_encoder_destroy(self.ptr)
            self.ptr = None


class OpusDecoder:
    def __init__(self, config: OpusConfig):
        self.config = config
        self.lib = _lib()
        err = ctypes.c_int()
        self.ptr = self.lib.opus_decoder_create(config.sample_rate, config.channels, ctypes.byref(err))
        if not self.ptr or err.value != OPUS_OK:
            raise RuntimeError(f"opus_decoder_create failed: {err.value}")

    def decode(self, payload: bytes | None, fec: bool = False) -> bytes:
        out = (ctypes.c_int16 * (self.config.frame_samples * self.config.channels))()
        if payload:
            data = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
            data_ptr = data
            length = len(payload)
        else:
            data_ptr = None
            length = 0
        n = self.lib.opus_decode(
            self.ptr, data_ptr, length, out, self.config.frame_samples, int(fec)
        )
        if n < 0:
            raise RuntimeError(f"opus_decode failed: {n}")
        return bytes(memoryview(out).cast("B")[: n * self.config.channels * 2])

    def close(self) -> None:
        if self.ptr:
            self.lib.opus_decoder_destroy(self.ptr)
            self.ptr = None
