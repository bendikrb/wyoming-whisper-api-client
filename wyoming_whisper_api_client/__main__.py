#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import WHISPER_LANGUAGES, OPENAI_DEFAULT_MODEL
from .handler import WhisperAPIEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        required=True,
        help="URL of whisper.cpp to use (http:// or https://). Set to 'openai' to use OpenAI API.",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--openai-api-key",
        required=False,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key to use for OpenAI API request (--api=openai)."
    )
    parser.add_argument(
        "-m",
        "--openai-model",
        required=False,
        default=OPENAI_DEFAULT_MODEL,
        help=f"ID of the OpenAI model to use. Only {OPENAI_DEFAULT_MODEL} is currently available.",
    )
    parser.add_argument(
        "-l",
        "--language",
        required=False,
        help="The language of the input audio in ISO-639-1 format.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=False,
        help="An optional text to guide the model's style or continue a previous audio segment.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        required=False,
        default=0.0,
        help="The sampling temperature, between 0 and 1.",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper-cpp",
                description="Faster Whisper transcription via its API",
                attribution=Attribution(
                    name="Michael Hansen",
                    url="https://github.com/synesthesiam",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="whisper.cpp",
                        description="whisper.cpp",
                        attribution=Attribution(
                            name="rhasspy wyoming faster whisper",
                            url="https://github.com/rhasspy/wyoming-faster-whisper",
                        ),
                        installed=True,
                        languages=WHISPER_LANGUAGES,
                        version="1.0",
                    ),
                ],
            ),
        ],
    )

    # Load converted whisper API

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            WhisperAPIEventHandler,
            wyoming_info,
            args,
            model_lock,
        )
    )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main(), debug=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
