
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZeroTTS Worker - Vietnamese safe edition
- Giữ nguyên giao thức JSONL cho server.js
- Chuẩn hóa tiếng Việt, số, ngày tháng
- Không cắt câu ở 160 ký tự
- Chia câu dài thành các đoạn
- Xử lý Unicode an toàn trên Windows
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "3")

import sys
import json
import re
import time
import hashlib
import traceback
import unicodedata
from pathlib import Path

import numpy as np
from zerotts import ZeroTTS, normalize_vi_text
from zerotts.chunking import (
    normalize_punctuation,
    chunk_text,
    clean_segment_punctuation,
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEFAULT_VOICE = "maichi"
THREADS = 3
CHUNK_SECONDS = 12
SILENCE_SECONDS = 0.12

CACHE_DIR = Path(__file__).resolve().parent / "data" / "tts-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_CANDIDATES = [
    os.environ.get("ZEROTTS_MODEL_PATH", ""),
    str(Path.home() / ".cache" / "zerotts"),
]

# --------------------------------------------------
# WINDOWS UTF-8 SAFE JSON OUTPUT
# --------------------------------------------------

def emit(payload):
    try:
        data = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str
        )
        sys.stdout.buffer.write((data + "\n").encode("ascii"))
        sys.stdout.buffer.flush()
    except Exception:
        # Không để lỗi ghi stdout làm chết worker
        pass


def log(message):
    try:
        text = str(message)
        text = text.encode(
            "utf-8", errors="backslashreplace"
        ).decode("utf-8")
        sys.stderr.buffer.write(
            (text + "\n").encode("utf-8", errors="backslashreplace")
        )
        sys.stderr.buffer.flush()
    except Exception:
        pass


def safe_string(value):
    if value is None:
        return ""
    return str(value).encode(
        "utf-8", errors="replace"
    ).decode("utf-8").strip()


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def resolve_model_source():
    for candidate in LOCAL_MODEL_CANDIDATES:
        if (
            candidate
            and os.path.isdir(candidate)
            and os.path.isfile(os.path.join(candidate, "config.json"))
        ):
            return candidate

    return "zeroweight-ai/ZeroTTS"


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def clean_unicode(text):
    text = safe_string(text)

    # Chuẩn hóa chữ Unicode, giữ nguyên dấu tiếng Việt
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFC", text)

    result = []

    for char in text:
        category = unicodedata.category(char)

        # Giữ chữ, dấu, số, dấu câu và khoảng trắng
        if category[0] in ("L", "M", "N", "P", "Z"):
            result.append(char)
        elif char in "\n\r\t":
            result.append(" ")

    text = "".join(result)
    text = re.sub(r"\s+", " ", text).strip()

    # Chỉ thu gọn dấu câu lặp, không sửa chữ cái
    text = re.sub(r"([.!?,])\1{2,}", r"\1\1", text)

    return text


def prepare_text(raw_text):
    text = clean_unicode(raw_text)

    if not text:
        return ""

    # Chuẩn hóa số, ngày tháng, giờ và chữ viết tắt tiếng Việt
    try:
        text = normalize_vi_text(text)
    except Exception as exc:
        log("[ZeroTTS] Text normalization fallback: " + safe_string(exc))

    text = normalize_punctuation(text)
    text = clean_unicode(text)

    if not re.search(r"[A-Za-zÀ-ỹĐđ]", text):
        return ""

    return text


def make_segments(text):
    if not text:
        return []

    try:
        segments = chunk_text(
            text,
            max_chunk_sec=CHUNK_SECONDS
        )

        cleaned = []

        for segment in segments:
            segment = clean_segment_punctuation(segment)
            segment = clean_unicode(segment)

            if segment:
                cleaned.append(segment)

        if cleaned:
            return cleaned

    except Exception as exc:
        log("[ZeroTTS] Chunk fallback: " + safe_string(exc))

    # Dự phòng nếu phiên bản thư viện không hỗ trợ chunking
    return [text]


# --------------------------------------------------
# AUDIO PROCESSING
# --------------------------------------------------

def polish_waveform(wav, sample_rate):
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)

    if arr.size == 0:
        return arr.reshape(1, 0)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    arr = arr - np.mean(arr)

    peak = float(np.max(np.abs(arr)))

    if peak > 0.90:
        arr *= 0.90 / peak

    fade_in = min(int(sample_rate * 0.012), arr.size // 4)
    fade_out = min(int(sample_rate * 0.020), arr.size // 4)

    if fade_in > 1:
        ramp = 0.5 * (
            1.0 - np.cos(
                np.linspace(0, np.pi, fade_in, dtype=np.float32)
            )
        )
        arr[:fade_in] *= ramp

    if fade_out > 1:
        ramp = 0.5 * (
            1.0 + np.cos(
                np.linspace(0, np.pi, fade_out, dtype=np.float32)
            )
        )
        arr[-fade_out:] *= ramp

    return arr.reshape(1, -1)


def synthesize_segments(tts, text, voice):
    segments = make_segments(text)

    if not segments:
        return None

    audio_parts = []
    silence = np.zeros(
        int(tts.sample_rate * SILENCE_SECONDS),
        dtype=np.float32
    )

    for index, segment in enumerate(segments):
        log(
            "[ZeroTTS] Đang đọc đoạn "
            + str(index + 1)
            + "/"
            + str(len(segments))
        )

        wav = tts.synthesize(
            segment,
            voice=voice,
            eoa_extra_frames=2
        )

        if wav is None:
            continue

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        if wav.size == 0:
            continue

        audio_parts.append(wav)

        if index < len(segments) - 1:
            audio_parts.append(silence)

    if not audio_parts:
        return None

    combined = np.concatenate(audio_parts)

    return polish_waveform(combined, int(tts.sample_rate))


# --------------------------------------------------
# MAIN WORKER
# --------------------------------------------------

def main():
    start_time = time.time()

    model_src = resolve_model_source()

    emit({
        "type": "status",
        "status": "loading",
        "model": model_src
    })

    try:
        log("[ZeroTTS] Đang nạp model...")

        tts = ZeroTTS.from_pretrained(
            model_src,
            intra_op_num_threads=THREADS
        )

        voices = tts.list_voices()

        if DEFAULT_VOICE in voices:
            default_voice = DEFAULT_VOICE
        else:
            default_voice = voices[0] if voices else None

        emit({
            "type": "ready",
            "model": model_src,
            "voice": default_voice,
            "voices": voices,
            "sampleRate": int(tts.sample_rate),
            "loadSec": round(time.time() - start_time, 2)
        })

        log("[ZeroTTS] Worker đã sẵn sàng. Giọng: " + str(default_voice))

    except Exception as exc:
        emit({
            "type": "error",
            "fatal": True,
            "error": safe_string(exc),
            "trace": traceback.format_exc()
        })
        return

    # Đọc stdin dưới dạng bytes để tránh lỗi mã hóa Windows
    for raw_line in sys.stdin.buffer:
        try:
            line = raw_line.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            req = json.loads(line)

            if not isinstance(req, dict):
                continue

            req_id = req.get("id")

            original_text = safe_string(req.get("text", ""))

            voice = safe_string(
                req.get("voice") or default_voice
            ).lower()

            if voice not in voices:
                voice = default_voice

            text = prepare_text(original_text)

            if not text:
                emit({
                    "type": "result",
                    "id": req_id,
                    "ok": False,
                    "error": "Empty or invalid Vietnamese text"
                })
                continue

            log("[ZeroTTS] Đang xử lý văn bản...")

            # v4: không dùng lại cache của bản cũ
            cache_key = hashlib.sha1(
                ("v4::" + voice + "::" + text).encode("utf-8")
            ).hexdigest()[:20]

            out_name = voice + "_" + cache_key + ".wav"
            out_path = CACHE_DIR / out_name

            if out_path.is_file() and out_path.stat().st_size > 200:
                emit({
                    "type": "result",
                    "id": req_id,
                    "ok": True,
                    "cached": True,
                    "voice": voice,
                    "file": out_name,
                    "url": "/tts-cache/" + out_name,
                    "synthSec": 0.0
                })
                continue

            synth_start = time.time()

            audio = synthesize_segments(
                tts,
                text,
                voice
            )

            if audio is None or audio.size == 0:
                emit({
                    "type": "result",
                    "id": req_id,
                    "ok": False,
                    "error": "No audio generated"
                })
                continue

            tts.save_audio(audio, str(out_path))

            synth_sec = round(time.time() - synth_start, 2)
            duration_sec = round(
                audio.shape[-1] / float(tts.sample_rate),
                2
            )

            emit({
                "type": "result",
                "id": req_id,
                "ok": True,
                "cached": False,
                "voice": voice,
                "file": out_name,
                "url": "/tts-cache/" + out_name,
                "synthSec": synth_sec,
                "durationSec": duration_sec
            })

        except Exception as exc:
            log("[ZeroTTS] Request error: " + safe_string(exc))
            log(traceback.format_exc())

            emit({
                "type": "result",
                "id": locals().get("req_id"),
                "ok": False,
                "error": safe_string(exc)
            })


if __name__ == "__main__":
    main()
