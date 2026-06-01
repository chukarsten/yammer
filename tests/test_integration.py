"""
Integration tests with real threading.

Unlike test_main.py (which mocks threading.Thread and calls closures from
pytest's plain thread), these tests let _ui_update_thread run for real and
drive callbacks from asyncio coroutines to reproduce the exact conditions of
a live session.

Primary regression guarded: page.update() must be called from the
yammer-ui-render thread, not from an asyncio thread.  When it was called
from an asyncio executor thread, Flutter would receive the patch but not
schedule a repaint frame, so chat bubbles appeared only on the next user
tap.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import sys

# flet is already mocked by test_main.py module-level code; ensure it stays
_ft = sys.modules.get("flet") or MagicMock()
_ft.run = MagicMock()
sys.modules["flet"] = _ft

import audio_loop  # noqa: E402
import main as _main  # noqa: E402


# ── AppHarness ────────────────────────────────────────────────────────────────

class AppHarness:
    """
    Runs main.main(page) with real threading (no Thread mock).
    Tracks every page.update() call and which thread made it.
    """

    def __init__(self):
        self.page = MagicMock()
        self.update_calls: list[threading.Thread] = []
        self._updated = threading.Event()

        def _track_update():
            self.update_calls.append(threading.current_thread())
            self._updated.set()

        self.page.update.side_effect = _track_update

        # mic_level.visible must start False so _level_poll_thread doesn't fire
        _ft.ProgressBar.return_value.visible = False

        _gesture_kw: dict = {}
        _ft.GestureDetector.side_effect = lambda **kw: (_gesture_kw.update(kw), MagicMock())[1]

        with patch.object(audio_loop, "set_message_callback") as _smc:
            with patch.object(audio_loop, "set_chunk_callback") as _scc:
                with patch.object(audio_loop, "set_vocab_callback") as _svc:
                    with patch.object(audio_loop, "start_session") as _ss:
                        with patch.object(audio_loop, "play_mp3_background"):
                            _main.main(self.page)

        self.add_message = _smc.call_args[0][0]
        self.on_chunk   = _scc.call_args[0][0]
        self.on_vocab   = _svc.call_args[0][0]
        self.on_status  = _ss.call_args[0][0]
        self.on_tap_down = _gesture_kw.get("on_tap_down")
        self.on_tap_up   = _gesture_kw.get("on_tap_up")

    def wait_for_update(self, timeout: float = 1.0) -> bool:
        fired = self._updated.wait(timeout=timeout)
        self._updated.clear()
        return fired

    def reset(self):
        self.update_calls.clear()
        self._updated.clear()


@pytest.fixture()
def app():
    audio_loop._history.clear()
    audio_loop._vocab.clear()
    audio_loop._on_message = None
    audio_loop._on_chunk = None
    audio_loop._on_vocab = None
    h = AppHarness()
    yield h
    # daemon threads die with the process; no teardown needed


# ── rendering-thread tests ────────────────────────────────────────────────────

def test_plain_thread_calls_page_update_directly(app):
    """
    _trigger_update() from a plain thread (Flet event handler / test) must
    call page.update() synchronously on the SAME thread — no event signalling.
    """
    app.reset()
    caller: list[threading.Thread] = []
    original = app.page.update.side_effect

    def capturing():
        caller.append(threading.current_thread())
        original()

    app.page.update.side_effect = capturing
    app.on_status("ready")

    assert caller, "page.update() was never called"
    assert caller[0] is threading.current_thread(), (
        "page.update() was not called synchronously from the calling thread"
    )


def test_asyncio_trigger_renders_via_ui_render_thread(app):
    """
    _trigger_update() from an asyncio coroutine must NOT call page.update()
    on the asyncio thread — it must hand off to yammer-ui-render.

    Regression: with run_in_executor the patch was sent but Flutter never
    scheduled a repaint because the call came from asyncio's executor pool.
    """
    app.reset()
    asyncio_thread_id: list[int] = []

    async def _trigger():
        asyncio_thread_id.append(threading.current_thread().ident)
        app.on_chunk("assistant", "Bonjour")

    asyncio.run(_trigger())

    assert app.wait_for_update(timeout=1.0), (
        "page.update() was not called within 1 s after asyncio on_chunk()"
    )

    last_caller = app.update_calls[-1]
    assert last_caller.ident != asyncio_thread_id[0], (
        f"page.update() came from the asyncio thread '{last_caller.name}' "
        "— Flutter will NOT repaint!"
    )
    assert last_caller.name == "yammer-ui-render", (
        f"Expected yammer-ui-render thread, got '{last_caller.name}'"
    )


def test_rapid_asyncio_triggers_are_coalesced(app):
    """
    20 rapid _trigger_update() calls from asyncio should produce far fewer
    page.update() calls (the 33 ms rate-limit coalesces them).
    """
    app.reset()

    async def _spam():
        for i in range(20):
            app.on_chunk("assistant", f"word {i}")

    asyncio.run(_spam())
    time.sleep(0.2)  # let background thread settle

    n = len(app.update_calls)
    assert n < 20, f"Expected coalescing, got {n} page.update() calls for 20 triggers"
    assert n >= 1, "Expected at least one page.update() call"


# ── conversation-flow tests ────────────────────────────────────────────────────

def test_status_cycle_does_not_raise(app):
    for state in ("connecting", "ready", "speaking", "listening", "processing", "ready"):
        app.on_status(state)


def test_streaming_turn_then_finalize(app):
    """Simulate a full assistant turn: stream chunks then finalize with add_message."""
    app.on_status("connecting")
    app.on_status("ready")

    for word in ("Bon", "Bonjour", "Bonjour !"):
        app.on_chunk("assistant", word)

    app.add_message("assistant", "Bonjour !", "**thinking**")
    assert app.wait_for_update(timeout=0.5)


def test_user_turn_after_assistant(app):
    """Full back-and-forth: assistant speaks, user speaks."""
    app.on_status("connecting")
    app.on_chunk("assistant", "Comment")
    app.on_chunk("assistant", "Comment allez-vous ?")
    app.add_message("assistant", "Comment allez-vous ?")

    app.on_status("listening")
    app.on_chunk("user", "Je")
    app.on_chunk("user", "Je vais bien")
    app.add_message("user", "Je vais bien")
    assert app.wait_for_update(timeout=0.5)


def test_reconnect_clears_streaming_state(app):
    """
    on_status('connecting') must clear _streaming so that the next on_chunk
    creates a fresh bubble rather than updating a stale (unmounted) widget.
    """
    app.on_chunk("assistant", "Bonjour")          # creates streaming bubble
    app.on_status("connecting")                    # must clear _streaming
    app.on_chunk("assistant", "Bonsoir")           # must NOT raise or update old widget
    assert app.wait_for_update(timeout=0.5)


def test_on_vocab_callback_fires_and_triggers_update(app):
    vocab = {
        "bonjour": {"listen": 3, "speak": 1, "translation": "Hello"},
        "merci":   {"listen": 2, "speak": 0, "translation": "Thank you"},
    }
    app.reset()
    app.on_vocab(vocab)
    assert app.wait_for_update(timeout=0.5)
