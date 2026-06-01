#!/usr/bin/env python
"""
Smoke test: verify chat bubbles appear immediately when on_chunk is called
from an asyncio context, WITHOUT requiring a mouse click.

Run:  python tests/smoke_test.py

Watch the chat panel on the right — roughly 3 seconds after the window opens,
a series of assistant bubbles should stream in ending with
"Bonjour ! Comment allez-vous ?" with no mouse interaction.

This guards against the regression where page.update() was sent from asyncio's
executor thread: Flutter received the patch but didn't schedule a repaint, so
bubbles only appeared on the next user tap.
"""
import asyncio
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_loop
import flet as ft

# ── prevent ft.run(main) from firing at import time ──────────────────────────

_run_patch = patch.object(ft, "run", MagicMock())
_run_patch.start()

import main as _main  # noqa: E402 — must come after ft.run is patched

_run_patch.stop()

# ── capture the callbacks that main() registers ───────────────────────────────

_on_status = [None]
_on_chunk = [None]
_add_message = [None]

_patches = [
    patch.object(audio_loop, "start_session",
                 side_effect=lambda cb: _on_status.__setitem__(0, cb)),
    patch.object(audio_loop, "set_chunk_callback",
                 side_effect=lambda cb: _on_chunk.__setitem__(0, cb)),
    patch.object(audio_loop, "set_message_callback",
                 side_effect=lambda cb: _add_message.__setitem__(0, cb)),
    patch.object(audio_loop, "set_vocab_callback"),
    patch.object(audio_loop, "play_mp3_background"),
]
for p in _patches:
    p.start()


# ── injector thread ───────────────────────────────────────────────────────────

def _injector():
    time.sleep(2.5)  # wait for the Flet window to fully open

    if _on_status[0] is None:
        print("[smoke] FAIL — on_status callback was never captured")
        return
    if _on_chunk[0] is None:
        print("[smoke] FAIL — on_chunk callback was never captured")
        return

    print("[smoke] App ready — running status cycle...")
    _on_status[0]("connecting")
    time.sleep(0.1)
    _on_status[0]("ready")
    time.sleep(0.5)

    # ── critical path: inject on_chunk from an asyncio context ────────────────
    # Before the _ui_update_thread fix, this path called page.update() from
    # asyncio's executor thread and Flutter never scheduled a repaint.
    print("[smoke] Injecting assistant chunks from asyncio context...")

    async def _chunks():
        words = ("Bon", "Bonjour", "Bonjour !", "Bonjour ! Comment allez-vous ?")
        for w in words:
            _on_chunk[0]("assistant", w)
            await asyncio.sleep(0.25)

    asyncio.run(_chunks())

    time.sleep(0.1)
    if _add_message[0]:
        _add_message[0]("assistant", "Bonjour ! Comment allez-vous ?", "**thinking**")

    print("[smoke] PASS — if the bubble is visible above, rendering works correctly.")
    print("[smoke] FAIL — if the chat panel is still empty, rendering is broken.")

    time.sleep(1.0)

    # ── second pass: user turn ────────────────────────────────────────────────
    print("[smoke] Simulating user turn...")
    _on_status[0]("listening")
    time.sleep(0.3)

    async def _user_chunks():
        _on_chunk[0]("user", "Je")
        await asyncio.sleep(0.2)
        _on_chunk[0]("user", "Je vais bien")

    asyncio.run(_user_chunks())
    time.sleep(0.1)
    if _add_message[0]:
        _add_message[0]("user", "Je vais bien")

    _on_status[0]("processing")
    time.sleep(0.5)
    _on_status[0]("ready")

    print("[smoke] All callbacks fired. Close the window to exit.")


threading.Thread(target=_injector, daemon=True).start()

# ── start the real app ────────────────────────────────────────────────────────

ft.run(_main.main)
