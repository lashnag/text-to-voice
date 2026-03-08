import io
import logging
import threading

import torch
import soundfile as sf

SAMPLE_RATE = 48000

# Silero поддерживает эти голоса для русского языка
RU_SPEAKERS = {"aidar", "baya", "kseniya", "xenia", "random"}
EN_SPEAKERS = {"en_0", "en_1", "en_2", "en_3"}

SUPPORTED_LANGUAGES = {"ru", "en"}

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
                    speaker="v4_ru",
                )
                _model = model
                logging.getLogger().info("Silero TTS model loaded")
    return _model

def get_default_speaker(language: str) -> str:
    if language == "en":
        return "en_0"
    return "xenia"

def validate_speaker(speaker: str, language: str):
    allowed = RU_SPEAKERS if language == "ru" else EN_SPEAKERS
    if speaker not in allowed:
        raise ValueError(
            f"Speaker '{speaker}' is not available for language '{language}'. "
            f"Available: {sorted(allowed)}"
        )

def synthesize(text: str, speaker: str, language: str) -> bytes:
    """
    Синтезирует речь и возвращает аудио в формате OGG/Opus.
    """
    model = get_model()

    logging.getLogger().info(
        f"Synthesizing: language={language}, speaker={speaker}, text_length={len(text)}"
    )

    with torch.no_grad():
        audio_tensor = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=SAMPLE_RATE,
            put_accent=True,
            put_yo=True,
        )

    from pydub import AudioSegment

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_tensor.numpy(), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)

    segment = AudioSegment.from_wav(wav_buf)
    ogg_buf = io.BytesIO()

    segment.export(ogg_buf, format="ogg", codec="libopus", parameters=["-vbr", "on"])
    ogg_buf.seek(0)

    return ogg_buf.read()
