import io
import logging
import re
import threading

import torch
import soundfile as sf
from pydub import AudioSegment
from transliterate import translit

SAMPLE_RATE = 48000
MAX_CHUNK_CHARS = 900

RU_SPEAKERS = {"aidar", "baya", "kseniya", "xenia", "random"}
EN_SPEAKERS = {"en_0", "en_1", "en_2", "en_3"}

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logging.getLogger().info("Loading Silero TTS model...")
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v5_2_ru",
                )
                _model = model
                logging.getLogger().info("Silero TTS model loaded")
    return _model


def get_default_speaker(language: str) -> str:
    return "en_0" if language == "en" else "aidar"


def validate_speaker(speaker: str, language: str):
    allowed = RU_SPEAKERS if language == "ru" else EN_SPEAKERS
    if speaker not in allowed:
        raise ValueError(
            f"Speaker '{speaker}' is not available for language '{language}'. "
            f"Available: {sorted(allowed)}"
        )


def _split_text(text: str) -> list[str]:
    chunks = []
    while len(text) > MAX_CHUNK_CHARS:
        split_pos = max(
            text.rfind('. ', 0, MAX_CHUNK_CHARS),
            text.rfind(', ', 0, MAX_CHUNK_CHARS),
            text.rfind(' ', 0, MAX_CHUNK_CHARS),
        )
        if split_pos == -1:
            split_pos = MAX_CHUNK_CHARS
        chunks.append(text[:split_pos + 1].strip())
        text = text[split_pos + 1:].strip()
    if text:
        chunks.append(text)
    return chunks


def _synthesize_chunk(model, text: str, speaker: str) -> AudioSegment:
    with torch.no_grad():
        audio_tensor = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=SAMPLE_RATE,
            put_accent=True,
            put_yo=True,
        )
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_tensor.numpy(), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)
    return AudioSegment.from_wav(wav_buf)


def synthesize(text: str, speaker: str, language: str) -> bytes:
    text = re.sub(r'[a-zA-Z]+', lambda m: translit(m.group(), 'ru'), text)

    logging.getLogger().info(
        f"Synthesizing: language={language}, speaker={speaker}, text_length={len(text)}"
    )

    model = get_model()
    segments = [_synthesize_chunk(model, chunk, speaker) for chunk in _split_text(text)]

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    ogg_buf = io.BytesIO()
    combined.export(ogg_buf, format="ogg", codec="libopus", parameters=["-vbr", "on"])
    ogg_buf.seek(0)
    return ogg_buf.read()