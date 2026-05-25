import pytest
import audio_loop


@pytest.fixture(autouse=True)
def reset_audio_loop_globals():
    """Restore all mutable module-level state between tests."""
    audio_loop._history.clear()
    audio_loop._on_message = None
    audio_loop._outgoing = None
    audio_loop._loop = None
    audio_loop._recording.clear()
    yield
    audio_loop._history.clear()
    audio_loop._on_message = None
    audio_loop._outgoing = None
    audio_loop._loop = None
    audio_loop._recording.clear()
