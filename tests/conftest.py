import sys
from unittest.mock import MagicMock

import pytest
import audio_loop

# ── single shared flet mock ───────────────────────────────────────────────────
# Must be installed before any test module imports main.py, otherwise each
# test file creates its own MagicMock() and main.py binds ft to whichever
# was first, causing fixtures in later-loaded files to configure the wrong mock.

if "flet" not in sys.modules:
    _ft = MagicMock()
    _ft.run = MagicMock()
    sys.modules["flet"] = _ft


@pytest.fixture(autouse=True)
def reset_audio_loop_globals():
    """Restore all mutable module-level state between tests."""
    audio_loop._history.clear()
    audio_loop._vocab.clear()
    audio_loop._on_message = None
    audio_loop._on_chunk = None
    audio_loop._on_vocab = None
    audio_loop._outgoing = None
    audio_loop._loop = None
    audio_loop._recording.clear()
    yield
    audio_loop._history.clear()
    audio_loop._vocab.clear()
    audio_loop._on_message = None
    audio_loop._on_chunk = None
    audio_loop._on_vocab = None
    audio_loop._outgoing = None
    audio_loop._loop = None
    audio_loop._recording.clear()
