import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from logger import init_logger, request_headers
from text_to_audio_service import get_default_speaker, get_model, synthesize, validate_speaker

init_logger()
logging.getLogger().info("TTS service run")

server = FastAPI()

get_model()

@server.post("/generate")
async def get_audio(request: Request):
    request_headers.set(dict(request.headers))
    try:
        body = await request.json()

        text: str = body.get("text", "").strip()
        language: str = body.get("language", "ru")
        speaker: str = body.get("speaker") or get_default_speaker(language)

        try:
            validate_speaker(speaker, language)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        audio_bytes = synthesize(text=text, speaker=speaker, language=language)
        return Response(content=audio_bytes, media_type="audio/ogg")

    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger().error(f"Common error: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при синтезе речи")


@server.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}
