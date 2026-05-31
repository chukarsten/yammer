"""
Full coverage tests for audio_loop.py.

Every function and reachable branch is exercised. External I/O (sounddevice,
pygame, Gemini network) is mocked; real asyncio primitives are used so
async semantics stay accurate.
"""

import asyncio
import queue
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import audio_loop
from audio_loop import CHUNK, CHANNELS, _END, _PLAYBACK_END


# ── helpers ──────────────────────────────────────────────────────────────────

def make_sc(model_turn=None, output_transcription=None,
            input_transcription=None, turn_complete=False):
    return SimpleNamespace(
        model_turn=model_turn,
        output_transcription=output_transcription,
        input_transcription=input_transcription,
        turn_complete=turn_complete,
    )


def make_model_turn(parts):
    return SimpleNamespace(parts=parts)


def make_part(inline_data=None, text=None):
    return SimpleNamespace(inline_data=inline_data, text=text)


def make_inline(data: bytes):
    return SimpleNamespace(data=data)


def make_session(messages):
    async def _gen():
        for m in messages:
            yield m

    session = MagicMock()
    session.receive = MagicMock(return_value=_gen())
    return session


def make_msg(sc=None):
    return SimpleNamespace(server_content=sc)


_real_sleep = asyncio.sleep


async def _fast_sleep(delay, result=None):
    """Drop-in that yields once but never actually waits — keeps test loops runnable."""
    await _real_sleep(0)


def _live_ctx(session):
    """Return a callable that acts as an async context manager, ignoring connect kwargs."""
    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield session
    return _ctx


# ── set_message_callback / set_chunk_callback ────────────────────────────────

def test_set_message_callback_stores_callable():
    cb = MagicMock()
    audio_loop.set_message_callback(cb)
    assert audio_loop._on_message is cb


def test_set_chunk_callback_stores_callable():
    cb = MagicMock()
    audio_loop.set_chunk_callback(cb)
    assert audio_loop._on_chunk is cb


# ── _add_message ─────────────────────────────────────────────────────────────

def test_add_message_strips_and_skips_blank():
    audio_loop._add_message("user", "   ")
    assert audio_loop._history == []


def test_add_message_appends_stripped_text():
    audio_loop._add_message("user", "  hello  ")
    assert audio_loop._history == [{"role": "user", "text": "hello"}]


def test_add_message_without_callback_does_not_raise():
    audio_loop._on_message = None
    audio_loop._add_message("assistant", "bonjour")
    assert len(audio_loop._history) == 1


def test_add_message_invokes_callback():
    received = []
    audio_loop._on_message = lambda role, text, thinking="": received.append((role, text))
    audio_loop._add_message("assistant", "bonjour")
    assert received == [("assistant", "bonjour")]


# ── play_mp3_background ──────────────────────────────────────────────────────

def test_play_mp3_background_calls_pygame():
    with patch("audio_loop.pygame.mixer.music") as m:
        audio_loop.play_mp3_background("/fake/track.mp3", volume=0.3)
    m.load.assert_called_once_with("/fake/track.mp3")
    m.set_volume.assert_called_once_with(0.3)
    m.play.assert_called_once_with(loops=-1)


# ── stop_recording ───────────────────────────────────────────────────────────

def test_stop_recording_clears_event():
    audio_loop._recording.set()
    audio_loop.stop_recording()
    assert not audio_loop._recording.is_set()


# ── start_session ─────────────────────────────────────────────────────────────

def test_start_session_executes_inner_run_closure():
    done = threading.Event()

    async def _quick(on_status):
        done.set()

    # Capture the _run closure without actually starting the thread.
    run_fn = [None]

    def fake_thread(*a, target=None, daemon=None, **kw):
        run_fn[0] = target
        return MagicMock()

    with patch("audio_loop.threading.Thread", side_effect=fake_thread):
        audio_loop.start_session(lambda s: None)

    assert run_fn[0] is not None

    # Now invoke _run in a real thread while _session_main is patched so the
    # global lookup inside _run resolves to our no-op.
    with patch.object(audio_loop, "_session_main", side_effect=_quick):
        t = threading.Thread(target=run_fn[0], daemon=True)
        t.start()
        assert done.wait(timeout=3.0), "_run closure never executed"


# ── _playback_worker ──────────────────────────────────────────────────────────

def test_playback_worker_plays_pcm_then_signals_ready():
    with patch("audio_loop.sd.OutputStream") as mock_cls:
        mock_stream = MagicMock()
        mock_cls.return_value = mock_stream

        q = queue.Queue()
        q.put(np.zeros(64, dtype=np.int16).tobytes())
        q.put(_PLAYBACK_END)

        statuses = []
        audio_loop._playback_worker(q, statuses.append)

    mock_stream.start.assert_called_once()
    mock_stream.write.assert_called_once()
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert statuses == ["ready"]


# ── _collect_response ─────────────────────────────────────────────────────────

async def test_collect_response_skips_message_with_no_sc():
    msgs = [make_msg(sc=None), make_msg(sc=make_sc(turn_complete=True))]
    statuses = []
    await audio_loop._collect_response(make_session(msgs), statuses.append)
    assert "ready" in statuses


async def test_collect_response_turn_complete_no_audio_calls_ready():
    sc = make_sc(turn_complete=True)
    statuses = []
    await audio_loop._collect_response(make_session([make_msg(sc)]), statuses.append)
    assert statuses == ["ready"]


async def test_collect_response_audio_chunk_starts_playback_thread():
    chunk = b"\x01\x02" * 50
    sc = make_sc(
        model_turn=make_model_turn([make_part(inline_data=make_inline(chunk))]),
        turn_complete=True,
    )
    statuses = []
    with patch("audio_loop.threading.Thread") as mock_thread:
        await audio_loop._collect_response(make_session([make_msg(sc)]), statuses.append)

    assert "speaking" in statuses
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()



async def test_collect_response_part_without_inline_data_skips_playback():
    sc = make_sc(
        model_turn=make_model_turn([make_part(inline_data=None)]),
        turn_complete=True,
    )
    statuses = []
    with patch("audio_loop.threading.Thread") as mock_thread:
        await audio_loop._collect_response(make_session([make_msg(sc)]), statuses.append)

    mock_thread.assert_not_called()
    assert "ready" in statuses


async def test_collect_response_output_transcription_added_to_history():
    sc1 = make_sc(output_transcription=SimpleNamespace(text="Bon"))
    sc2 = make_sc(output_transcription=SimpleNamespace(text="jour"), turn_complete=True)

    with patch("audio_loop.threading.Thread"):
        await audio_loop._collect_response(make_session([make_msg(sc1), make_msg(sc2)]),
                                            lambda s: None)

    assert any(h["role"] == "assistant" and "Bonjour" in h["text"]
               for h in audio_loop._history)


async def test_collect_response_output_transcription_calls_chunk_callback():
    chunks = []
    audio_loop._on_chunk = lambda role, text: chunks.append((role, text))
    sc1 = make_sc(output_transcription=SimpleNamespace(text="Bon"))
    sc2 = make_sc(output_transcription=SimpleNamespace(text="jour"), turn_complete=True)

    with patch("audio_loop.threading.Thread"):
        await audio_loop._collect_response(make_session([make_msg(sc1), make_msg(sc2)]),
                                            lambda s: None)

    assert chunks == [("assistant", "Bon"), ("assistant", "Bonjour")]


async def test_collect_response_thinking_text_not_shown_or_saved():
    """model_turn text parts are thinking/reasoning — must not reach UI or history."""
    chunks = []
    audio_loop._on_chunk = lambda role, text: chunks.append((role, text))
    sc = make_sc(
        model_turn=make_model_turn([make_part(text="**Thinking...**")]),
        output_transcription=SimpleNamespace(text="Bonjour!"),
        turn_complete=True,
    )
    await audio_loop._collect_response(make_session([make_msg(sc)]), lambda s: None)

    # Thinking text must NOT appear in chunks or history
    assert all("Thinking" not in t for _, t in chunks)
    assert not any(h["text"] == "**Thinking...**" for h in audio_loop._history)
    # But the real transcription must still appear
    assert ("assistant", "Bonjour!") in chunks


async def test_collect_response_thinking_passed_to_message_callback():
    """part.text (chain-of-thought) is collected and forwarded via _on_message."""
    received = []
    audio_loop._on_message = lambda role, text, thinking="": received.append((role, text, thinking))
    sc = make_sc(
        model_turn=make_model_turn([make_part(text="**Planning**")]),
        output_transcription=SimpleNamespace(text="Bonjour!"),
        turn_complete=True,
    )
    await audio_loop._collect_response(make_session([make_msg(sc)]), lambda s: None)

    assert len(received) == 1
    role, text, thinking = received[0]
    assert role == "assistant"
    assert text == "Bonjour!"
    assert "**Planning**" in thinking


async def test_collect_response_input_transcription_calls_chunk_callback():
    chunks = []
    audio_loop._on_chunk = lambda role, text: chunks.append((role, text))
    sc = make_sc(input_transcription=SimpleNamespace(text="Oui"), turn_complete=True)
    await audio_loop._collect_response(make_session([make_msg(sc)]), lambda s: None)
    assert ("user", "Oui") in chunks


async def test_collect_response_input_transcription_added_to_history():
    sc = make_sc(input_transcription=SimpleNamespace(text="Oui"), turn_complete=True)
    await audio_loop._collect_response(make_session([make_msg(sc)]), lambda s: None)
    assert any(h["role"] == "user" and h["text"] == "Oui" for h in audio_loop._history)


async def test_collect_response_turn_complete_with_audio_puts_sentinel():
    chunk = b"\x00\x01" * 50
    sc = make_sc(
        model_turn=make_model_turn([make_part(inline_data=make_inline(chunk))]),
        turn_complete=True,
    )
    put_calls = []
    with patch("audio_loop.queue.Queue") as mock_q_cls:
        mock_q = MagicMock()
        mock_q.put.side_effect = put_calls.append
        mock_q_cls.return_value = mock_q
        with patch("audio_loop.threading.Thread"):
            await audio_loop._collect_response(make_session([make_msg(sc)]), lambda s: None)

    assert _PLAYBACK_END in put_calls


async def test_collect_response_sc_without_transcription_attrs():
    # ServerContent with no output/input_transcription attributes at all
    sc = SimpleNamespace(model_turn=None, turn_complete=True)
    statuses = []
    await audio_loop._collect_response(make_session([make_msg(sc)]), statuses.append)
    assert "ready" in statuses


# ── _send_audio ───────────────────────────────────────────────────────────────

async def test_send_audio_calls_three_realtime_inputs():
    session = AsyncMock()
    await audio_loop._send_audio(session, b"\xAB\xCD" * 40)

    assert session.send_realtime_input.call_count == 3
    calls = session.send_realtime_input.call_args_list
    assert "activity_start" in calls[0].kwargs
    assert "audio" in calls[1].kwargs
    assert calls[1].kwargs["audio"].data == b"\xAB\xCD" * 40
    assert "activity_end" in calls[2].kwargs


# ── _session_main ─────────────────────────────────────────────────────────────

async def test_session_main_first_connect_sends_begin():
    session = AsyncMock()
    audio_loop._history.clear()

    with patch.object(audio_loop, "_collect_response", new=AsyncMock()):
        with patch.object(audio_loop._client.aio.live, "connect", new=_live_ctx(session)):
            task = asyncio.create_task(audio_loop._session_main(lambda s: None))
            await asyncio.sleep(0)
            await audio_loop._outgoing.put(_END)
            await asyncio.wait_for(task, timeout=2.0)

    text = session.send_client_content.call_args.kwargs["turns"].parts[0].text
    assert text == "begin"


async def test_session_main_reconnect_uses_system_instruction():
    """On reconnect, history goes into system_instruction — no second send_client_content."""
    audio_loop._history.clear()
    send_texts = []
    collect_count = [0]
    connect_configs = []

    async def mock_collect(s, on_status):
        collect_count[0] += 1
        if collect_count[0] == 1:
            audio_loop._history.append({"role": "assistant", "text": "Bonjour!"})
            raise Exception("simulated disconnect")

    session = AsyncMock()
    session.send_client_content = AsyncMock(
        side_effect=lambda **kw: send_texts.append(kw["turns"].parts[0].text)
    )

    reconnected = asyncio.Event()

    @asynccontextmanager
    async def capturing_ctx(*args, **kwargs):
        connect_configs.append(kwargs.get("config"))
        if len(connect_configs) >= 2:
            reconnected.set()
        yield session

    statuses = []
    with patch.object(audio_loop, "_collect_response", side_effect=mock_collect):
        with patch.object(audio_loop._client.aio.live, "connect", new=capturing_ctx):
            with patch("asyncio.sleep", new=_fast_sleep):
                task = asyncio.create_task(audio_loop._session_main(statuses.append))
                await asyncio.wait_for(reconnected.wait(), timeout=3.0)
                await audio_loop._outgoing.put(_END)
                await asyncio.wait_for(task, timeout=3.0)

    # Only one send_client_content ("begin") — reconnect doesn't send a second message
    assert send_texts == ["begin"]
    # Reconnect config has history in system_instruction
    assert len(connect_configs) >= 2
    reconnect_config = connect_configs[1]
    assert "CONVERSATION SO FAR:" in reconnect_config.system_instruction
    assert "Bonjour!" in reconnect_config.system_instruction
    # ready status fired on reconnect
    assert "ready" in statuses


async def test_session_main_drains_stale_queue_item():
    session = AsyncMock()
    stale_put = asyncio.Event()

    async def mock_collect_with_stale(s, on_status):
        await audio_loop._outgoing.put(b"\xff" * 100)
        stale_put.set()

    with patch.object(audio_loop, "_collect_response", side_effect=mock_collect_with_stale):
        with patch.object(audio_loop._client.aio.live, "connect", new=_live_ctx(session)):
            task = asyncio.create_task(audio_loop._session_main(lambda s: None))
            await asyncio.wait_for(stale_put.wait(), timeout=2.0)
            # drain has already run synchronously after mock_collect returned
            await audio_loop._outgoing.put(_END)
            await asyncio.wait_for(task, timeout=2.0)


async def test_session_main_processes_audio_turn():
    session = AsyncMock()
    collect_count = [0]

    async def mock_collect(s, on_status):
        collect_count[0] += 1

    with patch.object(audio_loop, "_collect_response", side_effect=mock_collect):
        with patch.object(audio_loop, "_send_audio", new=AsyncMock()):
            with patch.object(audio_loop._client.aio.live, "connect", new=_live_ctx(session)):
                task = asyncio.create_task(audio_loop._session_main(lambda s: None))
                await asyncio.sleep(0)
                await audio_loop._outgoing.put(b"\x01\x02" * 50)
                for _ in range(100):
                    await asyncio.sleep(0)
                    if collect_count[0] >= 2:
                        break
                await audio_loop._outgoing.put(_END)
                await asyncio.wait_for(task, timeout=2.0)

    assert collect_count[0] >= 2


async def test_session_main_exception_reconnects():
    call_count = [0]

    @asynccontextmanager
    async def mock_connect_fail_once(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ConnectionError("transient error")
        yield AsyncMock()

    statuses = []
    with patch.object(audio_loop, "_collect_response", new=AsyncMock()):
        with patch.object(audio_loop._client.aio.live, "connect",
                          new=mock_connect_fail_once):
            with patch("asyncio.sleep", new=_fast_sleep):
                task = asyncio.create_task(audio_loop._session_main(statuses.append))
                for _ in range(200):
                    await _real_sleep(0)
                    if call_count[0] >= 2:
                        break
                await audio_loop._outgoing.put(_END)
                await asyncio.wait_for(task, timeout=3.0)

    assert call_count[0] >= 2
    assert "connecting" in statuses


# ── record_and_stream ─────────────────────────────────────────────────────────

def _mock_input_stream(stop_after=1):
    reads = [0]
    data = np.zeros((CHUNK, CHANNELS), dtype=np.int16)

    def _read(n):
        reads[0] += 1
        if reads[0] >= stop_after:
            audio_loop._recording.clear()
        return data, None

    m = MagicMock()
    m.read.side_effect = _read
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


def _discard_coro(coro, loop):
    """Close the coroutine so it doesn't trigger 'never awaited' warnings."""
    coro.close()
    return MagicMock()


def test_record_and_stream_queues_audio():
    with patch("audio_loop.sd.InputStream", return_value=_mock_input_stream()):
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_discard_coro) as mock_rcts:
            audio_loop._outgoing = asyncio.Queue()
            audio_loop._loop = MagicMock()
            audio_loop.record_and_stream()

    mock_rcts.assert_called_once()
    assert mock_rcts.call_args[0][1] is audio_loop._loop


def test_record_and_stream_calls_on_level_callback():
    levels = []
    with patch("audio_loop.sd.InputStream", return_value=_mock_input_stream()):
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_discard_coro):
            audio_loop._outgoing = asyncio.Queue()
            audio_loop._loop = MagicMock()
            audio_loop.record_and_stream(on_level=levels.append)

    assert len(levels) >= 1
    assert all(0.0 <= v <= 1.0 for v in levels)


def test_record_and_stream_warns_when_loop_is_none():
    with patch("audio_loop.sd.InputStream", return_value=_mock_input_stream()):
        audio_loop._outgoing = asyncio.Queue()
        audio_loop._loop = None
        audio_loop.record_and_stream()  # should not raise; takes the warning branch


# ── _extract_words ────────────────────────────────────────────────────────────

def test_extract_words_basic():
    words = audio_loop._extract_words("Bonjour! Comment ça va?")
    assert "bonjour" in words
    assert "comment" in words
    assert "va" in words


def test_extract_words_strips_parenthetical():
    words = audio_loop._extract_words("Répétez (pause) s'il vous plaît")
    assert "pause" not in words
    assert "répétez" in words


def test_extract_words_handles_apostrophe():
    words = audio_loop._extract_words("c'est l'heure")
    assert "c'est" in words or "est" in words
    assert "l'heure" in words or "heure" in words


def test_extract_words_skips_single_char():
    words = audio_loop._extract_words("À la maison")
    for w in words:
        assert len(w) >= 2


# ── _update_vocab ─────────────────────────────────────────────────────────────

def test_update_vocab_listen_increments_for_assistant():
    # Simulate being past the first turn so vocab tracking is active
    audio_loop._history.append({"role": "assistant", "text": "greeting"})
    audio_loop._history.append({"role": "assistant", "text": "Bonjour tout le monde"})
    with patch("audio_loop.threading.Thread"):
        audio_loop._update_vocab("assistant", "Bonjour tout le monde")
    assert audio_loop._vocab["bonjour"]["listen"] == 1
    assert audio_loop._vocab["bonjour"]["speak"] == 0


def test_update_vocab_skips_first_assistant_turn():
    # First assistant message (len(_history) == 1) should NOT populate vocab
    audio_loop._history.append({"role": "assistant", "text": "Hello! I will help you"})
    with patch("audio_loop.threading.Thread") as mock_t:
        audio_loop._update_vocab("assistant", "Hello I will help you")
    assert audio_loop._vocab == {}
    mock_t.assert_not_called()


def test_update_vocab_speak_credits_known_words_only():
    # User gets speak credit only for words already in vocab
    audio_loop._vocab["bonjour"] = {"listen": 1, "speak": 0, "translation": "Hello"}
    audio_loop._update_vocab("user", "Bonjour madame")
    assert audio_loop._vocab["bonjour"]["speak"] == 1
    assert "madame" not in audio_loop._vocab  # new word from user not added


def test_update_vocab_speak_ignores_unknown_words():
    audio_loop._update_vocab("user", "hello world unknown")
    assert audio_loop._vocab == {}


def test_update_vocab_invokes_on_vocab_callback():
    received = []
    audio_loop._on_vocab = lambda v: received.append(dict(v))
    audio_loop._history.append({"role": "assistant", "text": "greeting"})
    audio_loop._history.append({"role": "assistant", "text": "Bonjour"})
    with patch("audio_loop.threading.Thread"):
        audio_loop._update_vocab("assistant", "Bonjour")
    assert len(received) >= 1
    assert "bonjour" in received[-1]


def test_update_vocab_spawns_translate_thread_for_new_words():
    audio_loop._history.append({"role": "assistant", "text": "greeting"})
    audio_loop._history.append({"role": "assistant", "text": "Bonjour"})
    with patch("audio_loop.threading.Thread") as mock_t:
        audio_loop._update_vocab("assistant", "Bonjour")
    mock_t.assert_called_once()
    assert mock_t.call_args.kwargs.get("daemon") is True


def test_update_vocab_no_new_thread_for_known_words():
    audio_loop._history.append({"role": "assistant", "text": "greeting"})
    audio_loop._history.append({"role": "assistant", "text": "Bonjour"})
    audio_loop._vocab["bonjour"] = {"listen": 1, "speak": 0, "translation": "Hello"}
    with patch("audio_loop.threading.Thread") as mock_t:
        audio_loop._update_vocab("assistant", "Bonjour")
    mock_t.assert_not_called()


# ── set_vocab_callback ────────────────────────────────────────────────────────

def test_set_vocab_callback_stores_callable():
    cb = MagicMock()
    audio_loop.set_vocab_callback(cb)
    assert audio_loop._on_vocab is cb


# ── _add_message triggers vocab update ───────────────────────────────────────

def test_add_message_triggers_vocab_update():
    # First assistant message is the English greeting — vocab stays empty
    with patch("audio_loop.threading.Thread"):
        audio_loop._add_message("assistant", "Hello welcome")
    assert audio_loop._vocab == {}

    # Second assistant message is French — vocab should populate
    with patch("audio_loop.threading.Thread"):
        audio_loop._add_message("assistant", "Bonjour")
    assert "bonjour" in audio_loop._vocab
