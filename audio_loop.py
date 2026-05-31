import asyncio
import logging
import os
import queue
import threading
from typing import Callable

import numpy as np
import pygame
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
pygame.mixer.init()

log = logging.getLogger("audio_loop")

MIC_RATE = 16000
PLAYBACK_RATE = 24000
CHANNELS = 1
CHUNK = 1024

_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options={"api_version": "v1alpha"},
)

SYSTEM_PROMPT = """\
You are a French language tutor named Yammer. You are speaking with a beginner English speaker \
learning French through conversation.

Speaking rules:
- Speak very slowly. Pause between each phrase.
- Use only simple, common French words the learner is likely to know.
- Keep every response to 1-3 short sentences.
- When introducing a new word, use known words to explain it in context immediately after.
- Never switch to English unless this is the very first message (onboarding).

First message only: warmly greet the learner in English, briefly explain you will teach them \
French through listening and speaking, then say your first French phrase slowly with a pause \
between each word.\
"""

_base_config: dict = dict(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    ),
)

if hasattr(types, "AudioTranscriptionConfig"):
    _base_config["output_audio_transcription"] = types.AudioTranscriptionConfig()
    _base_config["input_audio_transcription"] = types.AudioTranscriptionConfig()
    log.debug("Transcription enabled in config")

CONFIG = types.LiveConnectConfig(**_base_config)

# Conversation history: [{"role": "user"|"assistant", "text": str}]
_history: list[dict] = []
_on_message: Callable | None = None
_on_chunk: Callable | None = None

_recording = threading.Event()
_outgoing: asyncio.Queue | None = None
_loop: asyncio.AbstractEventLoop | None = None
_END = object()


def set_message_callback(cb: Callable):
    global _on_message
    _on_message = cb


def set_chunk_callback(cb: Callable):
    global _on_chunk
    _on_chunk = cb


def _add_message(role: str, text: str):
    text = text.strip()
    if not text:
        return
    _history.append({"role": role, "text": text})
    log.info("History [%s]: %s", role, text)
    if _on_message:
        _on_message(role, text)


def play_mp3_background(path: str, volume: float = 0.2):
    log.info("Starting background music: %s at volume %.1f", path, volume)
    pygame.mixer.music.load(path)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play(loops=-1)


def start_session(on_status: Callable):
    global _loop

    def _run():
        global _loop
        log.info("Session thread started")
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(_session_main(on_status))
        log.info("Session thread ended")

    threading.Thread(target=_run, daemon=True).start()


_PLAYBACK_END = object()


def _playback_worker(audio_q: queue.Queue, on_status: Callable):
    stream = sd.OutputStream(samplerate=PLAYBACK_RATE, channels=1, dtype="float32")
    stream.start()
    while True:
        item = audio_q.get()
        if item is _PLAYBACK_END:
            stream.stop()
            stream.close()
            log.info("Playback complete")
            on_status("ready")
            break
        pcm = np.frombuffer(item, dtype=np.int16).astype(np.float32) / 32768.0
        stream.write(pcm)


async def _collect_response(session, on_status: Callable):
    """Receive messages until turn_complete, playing audio and collecting transcripts."""
    playback_q: queue.Queue | None = None
    asst_transcript: list[str] = []
    user_transcript: list[str] = []

    async with asyncio.timeout(30):
        async for message in session.receive():
            sc = message.server_content
            if not sc:
                continue
            if sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if part.inline_data:
                        chunk = part.inline_data.data
                        log.debug("Audio chunk: %d bytes", len(chunk))
                        if playback_q is None:
                            log.info("First audio chunk — starting playback")
                            on_status("speaking")
                            playback_q = queue.Queue()
                            threading.Thread(
                                target=_playback_worker,
                                args=(playback_q, on_status),
                                daemon=True,
                            ).start()
                            if _on_chunk:
                                _on_chunk("assistant", "…")
                        playback_q.put(chunk)
                    if getattr(part, "text", None):
                        # Model thinking/reasoning — log for debugging but never display
                        log.debug("Model thinking: %s", part.text)

            out_t = getattr(sc, "output_transcription", None)
            if out_t and getattr(out_t, "text", None):
                asst_transcript.append(out_t.text)
                log.debug("Asst transcript chunk: %s", out_t.text)
                if _on_chunk:
                    _on_chunk("assistant", "".join(asst_transcript))

            in_t = getattr(sc, "input_transcription", None)
            if in_t and getattr(in_t, "text", None):
                user_transcript.append(in_t.text)
                log.debug("User transcript chunk: %s", in_t.text)
                if _on_chunk:
                    _on_chunk("user", "".join(user_transcript))

            if sc.turn_complete:
                log.info("Turn complete")
                if user_transcript:
                    _add_message("user", "".join(user_transcript))
                if asst_transcript:
                    _add_message("assistant", "".join(asst_transcript))
                if playback_q:
                    playback_q.put(_PLAYBACK_END)
                else:
                    on_status("ready")
                return


async def _session_main(on_status: Callable):
    global _outgoing

    _outgoing = asyncio.Queue()
    is_first = True

    while True:
        try:
            log.info("Connecting to Gemini Live...")
            on_status("connecting")
            async with _client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-latest", config=CONFIG
            ) as session:
                if is_first or not _history:
                    log.info("Sending initial greeting")
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text="begin")]),
                        turn_complete=True,
                    )
                    is_first = False
                else:
                    history_lines = "\n".join(
                        f"Yammer: {t['text']}" if t["role"] == "assistant" else f"Student: {t['text']}"
                        for t in _history[-30:]
                    )
                    log.info("Reconnecting — injecting %d history turns", len(_history))
                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=(
                                f"Previous conversation:\n{history_lines}\n\n"
                                "Continue the lesson. Do not re-introduce yourself."
                            ))],
                        ),
                        turn_complete=True,
                    )

                await _collect_response(session, on_status)

                # Discard audio queued during connect/reconnect
                drained = 0
                while True:
                    try:
                        _outgoing.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    log.info("Drained %d stale audio item(s) from queue", drained)

                while True:
                    audio_bytes = await _outgoing.get()
                    if audio_bytes is _END:
                        return
                    await _send_audio(session, audio_bytes)
                    await _collect_response(session, on_status)

        except Exception as e:
            log.error("Session error: %s — reconnecting in 2s", e)
            on_status("connecting")
            await asyncio.sleep(2)


async def _send_audio(session, audio_bytes: bytes):
    log.info("Sending %d bytes — ActivityStart + audio + ActivityEnd", len(audio_bytes))
    await session.send_realtime_input(activity_start=types.ActivityStart())
    await session.send_realtime_input(
        audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
    )
    await session.send_realtime_input(activity_end=types.ActivityEnd())
    log.info("Audio sent — collecting response")


def record_and_stream(on_level: Callable = None):
    log.info("Opening microphone")
    _recording.set()
    chunks = []

    with sd.InputStream(samplerate=MIC_RATE, channels=CHANNELS, dtype="int16") as stream:
        while _recording.is_set():
            data, _ = stream.read(CHUNK)
            chunks.append(data.tobytes())
            if on_level:
                rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                on_level(min(rms / 1000.0, 1.0))

    audio_bytes = b"".join(chunks)
    log.info("Microphone closed — %d bytes captured", len(audio_bytes))

    if _outgoing and _loop and audio_bytes:
        asyncio.run_coroutine_threadsafe(_outgoing.put(audio_bytes), _loop)
        log.info("Audio queued for send_loop")
    else:
        log.warning("Cannot queue audio — outgoing=%s loop=%s bytes=%d",
                    _outgoing, _loop, len(audio_bytes))


def stop_recording():
    log.info("stop_recording called")
    _recording.clear()
