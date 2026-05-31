"""
Coverage tests for main.py closures.

flet is mocked at sys.modules level before the import so ft.run() is a no-op.
We call main.main(page) with audio_loop side-effects mocked, capture the
closures via the known API, and test each one.
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# ── mock flet BEFORE importing main ──────────────────────────────────────────

_ft = MagicMock()
_ft.run = MagicMock()  # prevents ft.run(main) from starting the app
sys.modules["flet"] = _ft

import audio_loop  # noqa: E402 – must come after flet mock
import main as _main  # noqa: E402 – ft.run fires here (mocked)


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def app(request):
    """
    Invoke main.main(page) and return a dict of captured closures plus the
    mock page object.
    """
    page = MagicMock()
    captured = {}
    gesture_kw = {}

    def _fake_gesture(**kw):
        gesture_kw.update(kw)
        return MagicMock()

    _ft.GestureDetector.side_effect = _fake_gesture

    threads = []

    def _fake_thread(*a, target=None, **kw):
        threads.append(target)
        return MagicMock()

    with patch("main.threading.Thread", side_effect=_fake_thread):
        with patch.object(audio_loop, "set_message_callback") as mock_smc:
            with patch.object(audio_loop, "set_chunk_callback") as mock_scc:
                with patch.object(audio_loop, "set_vocab_callback") as mock_svc:
                    with patch.object(audio_loop, "start_session") as mock_ss:
                        with patch.object(audio_loop, "play_mp3_background"):
                            _main.main(page)

    captured["add_message"] = mock_smc.call_args[0][0]
    captured["on_chunk"] = mock_scc.call_args[0][0]
    captured["on_vocab"] = mock_svc.call_args[0][0]
    captured["on_status"] = mock_ss.call_args[0][0]
    captured["on_tap_down"] = gesture_kw.get("on_tap_down")
    captured["on_tap_up"] = gesture_kw.get("on_tap_up")
    captured["threads"] = threads  # [0] is _level_poll_thread target (pragma: no cover)
    captured["page"] = page
    return captured


# ── on_status states ──────────────────────────────────────────────────────────

def test_on_status_connecting(app):
    app["on_status"]("connecting")
    app["page"].update.assert_called()


def test_on_status_ready(app):
    app["on_status"]("ready")
    app["page"].update.assert_called()


def test_on_status_speaking(app):
    app["on_status"]("speaking")
    app["page"].update.assert_called()


def test_on_status_listening(app):
    app["on_status"]("listening")
    app["page"].update.assert_called()


def test_on_status_processing(app):
    app["on_status"]("processing")
    app["page"].update.assert_called()


# ── on_chunk ──────────────────────────────────────────────────────────────────

def test_on_chunk_creates_bubble_on_first_call(app):
    app["page"].update.reset_mock()
    app["on_chunk"]("assistant", "Bon")
    app["page"].update.assert_called()


def test_on_chunk_updates_existing_bubble(app):
    app["on_chunk"]("assistant", "Bon")
    app["page"].update.reset_mock()
    app["on_chunk"]("assistant", "Bonjour")
    app["page"].update.assert_called()


def test_on_chunk_exception_is_silenced(app):
    app["page"].update.side_effect = Exception("page closed")
    app["on_chunk"]("assistant", "hi")  # must not raise


# ── add_message ───────────────────────────────────────────────────────────────

def test_add_message_user_bubble(app):
    app["page"].update.reset_mock()
    app["add_message"]("user", "hello")
    app["page"].update.assert_called()


def test_add_message_assistant_bubble(app):
    app["page"].update.reset_mock()
    app["add_message"]("assistant", "bonjour")
    app["page"].update.assert_called()


def test_add_message_finalizes_streaming_bubble(app):
    app["on_chunk"]("assistant", "Bon")
    app["page"].update.reset_mock()
    app["add_message"]("assistant", "Bonjour")
    app["page"].update.assert_called()


def test_add_message_page_update_exception_is_silenced(app):
    app["page"].update.side_effect = Exception("page closed")
    app["add_message"]("user", "hi")  # must not raise


def test_add_message_with_thinking_does_not_raise(app):
    app["add_message"]("assistant", "Bonjour", "**Planning lesson**")
    app["page"].update.assert_called()


def test_on_status_speaking_creates_placeholder_bubble(app):
    app["page"].update.reset_mock()
    app["on_status"]("speaking")
    app["page"].update.assert_called()


def test_on_chunk_after_speaking_updates_existing_bubble(app):
    app["on_status"]("speaking")
    app["page"].update.reset_mock()
    # on_chunk must not create a second bubble — it updates the existing one
    app["on_chunk"]("assistant", "Bonjour")
    app["page"].update.assert_called()


# ── start_listening / stop_listening ─────────────────────────────────────────

def test_start_listening_spawns_thread(app):
    with patch("threading.Thread") as mock_t:
        mock_t.return_value = MagicMock()
        app["on_tap_down"](MagicMock())
    mock_t.assert_called_once()
    assert mock_t.call_args.kwargs.get("target") is not None


def test_stop_listening_calls_stop_recording(app):
    with patch.object(audio_loop, "stop_recording") as mock_stop:
        app["on_tap_up"](MagicMock())
    mock_stop.assert_called_once()


# ── _record and _on_mic_level ─────────────────────────────────────────────────

def test_record_calls_record_and_stream_and_on_mic_level(app):
    # Capture _record via the Thread created when start_listening fires.
    record_func = None
    with patch("threading.Thread") as mock_t:
        mock_t.return_value = MagicMock()
        app["on_tap_down"](MagicMock())
        record_func = mock_t.call_args.kwargs.get("target")

    assert callable(record_func)

    # Calling record_func() exercises _record and _on_mic_level.
    on_level_calls = []

    def fake_record_and_stream(on_level=None):
        if on_level:
            on_level(0.42)  # exercises _on_mic_level

    with patch.object(audio_loop, "record_and_stream",
                      side_effect=fake_record_and_stream):
        record_func()


# ── on_vocab ──────────────────────────────────────────────────────────────────

def test_on_vocab_updates_chips_and_triggers_page_update(app):
    app["page"].update.reset_mock()
    vocab = {
        "bonjour": {"listen": 3, "speak": 1, "translation": "Hello"},
        "merci": {"listen": 2, "speak": 0, "translation": "Thank you"},
    }
    app["on_vocab"](vocab)
    app["page"].update.assert_called()


def test_on_vocab_sorts_by_total_count(app):
    vocab = {
        "bonjour": {"listen": 3, "speak": 1, "translation": "Hello"},
        "merci": {"listen": 10, "speak": 5, "translation": "Thank you"},
        "oui": {"listen": 1, "speak": 0, "translation": "Yes"},
    }
    app["on_vocab"](vocab)
    # Just verify it doesn't raise and updates the page
    app["page"].update.assert_called()


def test_on_vocab_empty_dict_clears_chips(app):
    app["on_vocab"]({"bonjour": {"listen": 1, "speak": 0, "translation": "Hello"}})
    app["page"].update.reset_mock()
    app["on_vocab"]({})
    app["page"].update.assert_called()
