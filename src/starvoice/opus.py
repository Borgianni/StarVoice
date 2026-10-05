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
    lib.opus_encoder_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int32,
    ]

    lib.opus_packet_get_samples_per_frame.restype = ctypes.c_int
    lib.opus_packet_get_samples_per_frame.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int32,
    ]
    lib.opus_packet_get_nb_channels.restype = ctypes.c_int
    lib.opus_packet_get_nb_channels.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.opus_packet_parse.restype = ctypes.c_int
    lib.opus_packet_parse.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int),
    ]

    try:
        native_lbrr = lib.opus_packet_has_lbrr
    except AttributeError:
        native_lbrr = None
    if native_lbrr is not None:
        native_lbrr.restype = ctypes.c_int
        native_lbrr.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
        ]

    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
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
            config.sample_rate,
            config.channels,
            OPUS_APPLICATION_VOIP,
            ctypes.byref(err),
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
        self._ctl(
            OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            expected_loss_percent if enabled else 0,
        )

    def set_dred_duration(self, frames_10ms: int) -> bool:
        rc = self.lib.opus_encoder_ctl(
            self.ptr,
            OPUS_SET_DRED_DURATION_REQUEST,
            ctypes.c_int(frames_10ms),
        )
        return rc == OPUS_OK

    def encode(self, pcm: bytes) -> bytes:
        expected = self.config.frame_samples * self.config.channels * 2
        if len(pcm) != expected:
            raise ValueError(f"expected {expected} PCM bytes, got {len(pcm)}")
        samples = (ctypes.c_int16 * (len(pcm) // 2)).from_buffer_copy(pcm)
        out = (ctypes.c_ubyte * 4000)()
        n = self.lib.opus_encode(
            self.ptr,
            samples,
            self.config.frame_samples,
            out,
            len(out),
        )
        if n < 0:
            raise RuntimeError(f"opus_encode failed: {n}")
        return bytes(out[:n])

    def close(self) -> None:
        if self.ptr:
            self.lib.opus_encoder_destroy(self.ptr)
            self.ptr = None


def _packet_has_fec_compat(lib: ctypes.CDLL, data, length: int) -> bool:
    """Compatibility implementation of opus_packet_has_lbrr for pre-1.5 libopus.

    Mirrors the public libopus 1.5 implementation using packet parser APIs that
    are available in older libopus releases.
    """
    frame_size = lib.opus_packet_get_samples_per_frame(data, 48000)
    if frame_size <= 0:
        raise RuntimeError(f"opus_packet_get_samples_per_frame failed: {frame_size}")

    nb_frames = frame_size // 960 if frame_size > 960 else 1
    channels = lib.opus_packet_get_nb_channels(data)
    if channels <= 0:
        raise RuntimeError(f"opus_packet_get_nb_channels failed: {channels}")

    frames = (ctypes.POINTER(ctypes.c_ubyte) * 48)()
    sizes = (ctypes.c_int16 * 48)()
    ret = lib.opus_packet_parse(
        data,
        length,
        None,
        frames,
        sizes,
        None,
    )
    if ret <= 0:
        raise RuntimeError(f"opus_packet_parse failed: {ret}")
    if sizes[0] == 0:
        return False

    first = frames[0][0]
    lbrr = (first >> (7 - nb_frames)) & 0x1
    if channels == 2:
        lbrr = lbrr or ((first >> (6 - 2 * nb_frames)) & 0x1)
    return bool(lbrr)


def packet_has_fec(payload: bytes) -> bool:
    if not payload:
        return False
    lib = _lib()
    data = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)

    try:
        native = lib.opus_packet_has_lbrr
    except AttributeError:
        native = None

    if native is not None:
        rc = native(data, len(payload))
        if rc < 0:
            raise RuntimeError(f"opus_packet_has_lbrr failed: {rc}")
        return bool(rc)

    return _packet_has_fec_compat(lib, data, len(payload))


class OpusDecoder:
    def __init__(self, config: OpusConfig):
        self.config = config
        self.lib = _lib()
        err = ctypes.c_int()
        self.ptr = self.lib.opus_decoder_create(
            config.sample_rate,
            config.channels,
            ctypes.byref(err),
        )
        if not self.ptr or err.value != OPUS_OK:
            raise RuntimeError(f"opus_decoder_create failed: {err.value}")

    def decode(self, payload: bytes | None, fec: bool = False) -> bytes:
        out = (
            ctypes.c_int16
            * (self.config.frame_samples * self.config.channels)
        )()
        if payload:
            data = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
            data_ptr = data
            length = len(payload)
        else:
            data_ptr = None
            length = 0
        n = self.lib.opus_decode(
            self.ptr,
            data_ptr,
            length,
            out,
            self.config.frame_samples,
            int(fec),
        )
        if n < 0:
            raise RuntimeError(f"opus_decode failed: {n}")
        return bytes(
            memoryview(out).cast("B")[
                : n * self.config.channels * 2
            ]
        )

    def close(self) -> None:
        if self.ptr:
            self.lib.opus_decoder_destroy(self.ptr)
            self.ptr = None
