"""
Iterative Gemini Live audio test. Records 3s of real mic audio then tries
multiple approaches to get a response. Run with: uv run test_gemini.py
"""
import asyncio
import logging
import os
import sounddevice as sd
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("test")
logging.getLogger("websockets").setLevel(logging.WARNING)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options={"api_version": "v1alpha"})

SPEECH_CFG = types.SpeechConfig(voice_config=types.VoiceConfig(
    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")))

def record_mic(seconds=3, rate=16000) -> bytes:
    log.info("Recording %ds of mic audio — speak now!", seconds)
    data = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    log.info("Recording done")
    return data.tobytes()

def silence(seconds=1.5, rate=16000) -> bytes:
    return bytes(int(seconds * rate) * 2)

async def collect(session, timeout=12.0) -> tuple[int, bool]:
    audio, done = 0, False
    try:
        async with asyncio.timeout(timeout):
            async for msg in session.receive():
                sc = msg.server_content
                if sc:
                    if sc.model_turn and sc.model_turn.parts:
                        for p in sc.model_turn.parts:
                            if p.inline_data:
                                audio += len(p.inline_data.data)
                    if sc.turn_complete:
                        done = True
                        break
    except TimeoutError:
        pass
    return audio, done

async def run(label, model, config, send_fn, speech):
    log.info("\n--- %s [%s] ---", label, model.split("/")[-1])
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            # Warm up: text greeting
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text="Say 'ok' only.")]),
                turn_complete=True)
            a, ok = await collect(session, timeout=10)
            if not ok:
                log.info("SKIP — warmup failed (%d bytes)", a)
                return False
            log.info("Warmup ok (%d bytes audio)", a)

            await send_fn(session, speech)
            a, ok = await collect(session, timeout=12)
            log.info("RESULT: %d bytes, complete=%s => %s", a, ok, "SUCCESS ✓" if (a > 0 and ok) else "FAIL ✗")
            return a > 0 and ok
    except Exception as e:
        log.info("ERROR: %s", e)
        return False

async def main():
    speech = record_mic(3)
    sil = silence(1.5)

    approaches = [
        # (label, model, config, send_fn)
        (
            "realtime_input + silence pad, VAD on",
            "models/gemini-2.5-flash-native-audio-latest",
            types.LiveConnectConfig(response_modalities=["AUDIO"], speech_config=SPEECH_CFG),
            lambda s, sp: s.send_realtime_input(
                audio=types.Blob(data=sp + sil, mime_type="audio/pcm;rate=16000")),
        ),
        (
            "realtime_input ActivityStart/End, VAD off",
            "models/gemini-2.5-flash-native-audio-latest",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"], speech_config=SPEECH_CFG,
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True))),
            lambda s, sp: _send_with_activity(s, sp),
        ),
        (
            "realtime_input + silence pad, VAD on [gemini-3.1-flash-live]",
            "models/gemini-3.1-flash-live-preview",
            types.LiveConnectConfig(response_modalities=["AUDIO"], speech_config=SPEECH_CFG),
            lambda s, sp: s.send_realtime_input(
                audio=types.Blob(data=sp + sil, mime_type="audio/pcm;rate=16000")),
        ),
        (
            "realtime_input ActivityStart/End, VAD off [gemini-3.1-flash-live]",
            "models/gemini-3.1-flash-live-preview",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"], speech_config=SPEECH_CFG,
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True))),
            lambda s, sp: _send_with_activity(s, sp),
        ),
    ]

    async def _send_with_activity(session, sp):
        await session.send_realtime_input(activity_start=types.ActivityStart())
        await session.send_realtime_input(
            audio=types.Blob(data=sp, mime_type="audio/pcm;rate=16000"))
        await session.send_realtime_input(activity_end=types.ActivityEnd())

    for label, model, config, send_fn in approaches:
        result = await run(label, model, config, send_fn, speech)
        if result:
            log.info("\n*** WINNING APPROACH: %s ***", label)
            break

asyncio.run(main())
