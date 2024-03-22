"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
import wave
from io import BytesIO

import httpx
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import OPENAI_API_URL

_LOGGER = logging.getLogger(__name__)


class WhisperAPIEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_lock = model_lock
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self._language = self.cli_args.language

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")

            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)
            self.audio += chunk.audio

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            async with httpx.AsyncClient() as client:
                with BytesIO() as tmpfile:
                    with wave.open(tmpfile, 'wb') as wavfile:
                        wavfile.setparams((
                            self.audio_converter.channels,
                            self.audio_converter.width,
                            self.audio_converter.rate,
                            0,
                            'NONE',
                            'NONE',
                        ))
                        wavfile.writeframes(self.audio)

                        request_args = {
                            "files": {
                                "file": ('speech.wav', tmpfile.getvalue(), 'audio/x-wav'),
                            },
                            "data": {
                                "temperature": self.cli_args.temperature,
                                "response_format": "json",
                                "prompt": self.cli_args.prompt,
                                "language": self._language,
                            },
                            "timeout": 120.0,
                        }
                        api_url = self.cli_args.api

                        if api_url == "openai":
                            api_url = OPENAI_API_URL
                            request_args["data"].update({
                                "model": self.cli_args.openai_model,
                                "temperature_inc": None,
                            })
                            request_args["headers"] = {
                                "Authorization": f"Bearer {self.cli_args.openai_api_key}",
                            }

                        request_args["data"] = {k: v for k, v in request_args["data"].items() if v is not None}

                        async with self.model_lock:
                            r = await client.post(api_url, **request_args)
                            text = r.json()['text']

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self.audio = bytes()
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
