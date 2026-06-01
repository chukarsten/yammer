"""
Integration tests with real threading.

Drive callbacks from asyncio coroutines to reproduce the exact conditions of
a live session.

Primary regression guarded: page.update() must be called via page.run_task(),
not directly from background threads.  page.run_task() sets Flet's internal
_context_page ContextVar and schedules on the Flet connection loop — the only
path that reliably triggers Flutter repaints for structural changes like new
chat bubbles.
"""

import asyncio
import threading
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

    page.run_task is wired to execute the coroutine synchronously on the
    calling thread so that page.update() is reachable in tests without a
    real Flet event loop.  In production, page.run_task() schedules the
    coroutine on the Flet connection loop (different thread, correct context).
    """

    def __init__(self):
        self.page = MagicMock()
        self.update_calls: list[threading.Thread] = []
        self._updated = threading.Event()

        def _track_update():
            self.update_calls.append(threading.current_thread())
            self._updated.set()

        self.page.update.side_effect = _track_update

        # Execute the async handler synchronously on the calling thread.
        # When called from within asyncio.run() on Windows a nested
        # ProactorEventLoop can't be started — close the coroutine to
        # suppress the "never awaited" warning and let the test check
        # run_task was called rather than waiting for page.update().
        def _mock_run_task(handler, *args, **kwargs):
            import asyncio
            coro = handler(*args, **kwargs)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            except Exception:
                coro.close()
            finally:
                loop.close()

        self.page.run_task.side_effect = _mock_run_task

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

def test_trigger_update_calls_page_run_task(app):
    """
    _trigger_update() must always call page.run_task() (not page.update()
    directly).  page.run_task schedules the update on the Flet connection
    loop, which sets _context_page before flushing — the only path that
    reliably triggers a Flutter repaint for structural changes like new
    chat bubbles.
    """
    app.reset()
    app.page.run_task.reset_mock()
    app.on_status("ready")

    app.page.run_task.assert_called(), "page.run_task() was not called"
    assert app.wait_for_update(timeout=1.0), (
        "page.update() was not called within 1 s after on_status()"
    )


def test_asyncio_trigger_calls_page_run_task(app):
    """
    _trigger_update() from an asyncio coroutine must call page.run_task()
    with an async function (not call page.update() directly on the asyncio
    thread).

    Regression guard: before the fix, page.update() was called via
    run_in_executor from the asyncio thread — Flutter received the diff but
    never scheduled a repaint, so bubbles only appeared on next tap.

    We check the API contract (run_task called with async fn) rather than
    waiting for page.update(), because running a second event loop inside
    asyncio.run() isn't possible on Windows.
    """
    import inspect

    app.page.run_task.reset_mock()

    async def _trigger():
        app.on_chunk("assistant", "Bonjour")

    asyncio.run(_trigger())

    app.page.run_task.assert_called()
    handler = app.page.run_task.call_args[0][0]
    assert inspect.iscoroutinefunction(handler), (
        f"Expected async function passed to run_task, got {handler!r}"
    )


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
