#!/usr/bin/env python
"""
Smoke test: verify chat bubbles appear immediately when on_chunk is called
from an asyncio context, WITHOUT requiring a mouse click.

Run:  python tests/smoke_test.py

Watch the chat panel on the right — roughly 3 seconds after the window opens,
a series of assistant bubbles should stream in ending with
"Bonjour ! Comment allez-vous ?" with no mouse interaction.

The test prints diagnostics so you can tell whether:
  - the callbacks were captured correctly
  - page.update() is being called and from which thread
  - the issue is plain-thread vs asyncio injection
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

import main as _main  # noqa: E402

_run_patch.stop()

# ── references we'll fill in once main(page) executes ────────────────────────

_on_status = [None]
_on_chunk = [None]
_add_message = [None]
_page_ref = [None]

# ── wrap main to capture the page object ─────────────────────────────────────

_original_main = _main.main


def _wrapped_main(page):
    _page_ref[0] = page

    # Wrap page.update so we can log every call and from which thread
    _real_update = page.update
    _count = [0]

    def _tracked_update(*args, **kwargs):
        _count[0] += 1
        print(f"[smoke] page.update() #{_count[0]} from thread '{threading.current_thread().name}'")
        return _real_update(*args, **kwargs)

    page.update = _tracked_update
    _original_main(page)


# ── patch audio_loop before ft.run calls main() ──────────────────────────────

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
    time.sleep(2.5)

    print(f"\n[smoke] on_status captured : {_on_status[0] is not None}")
    print(f"[smoke] on_chunk  captured : {_on_chunk[0] is not None}")
    print(f"[smoke] page      captured : {_page_ref[0] is not None}")

    if _on_status[0] is None or _on_chunk[0] is None:
        print("[smoke] FAIL — callbacks not captured; did main() run?")
        return

    # ── status cycle (plain thread → direct page.update()) ───────────────────
    print("\n[smoke] === Phase 1: status cycle (plain thread) ===")
    _on_status[0]("connecting")
    time.sleep(0.1)
    _on_status[0]("ready")
    time.sleep(0.5)

    # ── plain-thread injection ────────────────────────────────────────────────
    print("\n[smoke] === Phase 2: on_chunk from plain thread ===")
    print("[smoke] This path calls page.update() directly.")
    _on_chunk[0]("assistant", "Bonjour (plain thread)")
    time.sleep(0.8)
    print("[smoke] → If a bubble appeared above, plain-thread path works.")

    # ── asyncio injection ─────────────────────────────────────────────────────
    print("\n[smoke] === Phase 3: on_chunk from asyncio context ===")
    print("[smoke] This path signals _ui_update_thread to call page.update().")

    async def _chunks():
        words = ("Bon", "Bonjour", "Bonjour !", "Bonjour ! Comment allez-vous ?")
        for w in words:
            _on_chunk[0]("assistant", w)
            await asyncio.sleep(0.25)

    asyncio.run(_chunks())
    time.sleep(0.5)
    print("[smoke] → If the bubble updated above, asyncio path works.")

    # ── finalize with add_message ─────────────────────────────────────────────
    if _add_message[0]:
        _add_message[0]("assistant", "Bonjour ! Comment allez-vous ?", "**thinking**")
        time.sleep(0.3)

    # ── user turn ─────────────────────────────────────────────────────────────
    print("\n[smoke] === Phase 4: user turn ===")
    _on_status[0]("listening")
    time.sleep(0.3)

    async def _user():
        _on_chunk[0]("user", "Je")
        await asyncio.sleep(0.2)
        _on_chunk[0]("user", "Je vais bien")

    asyncio.run(_user())
    time.sleep(0.1)
    if _add_message[0]:
        _add_message[0]("user", "Je vais bien")
    _on_status[0]("processing")
    time.sleep(0.3)
    _on_status[0]("ready")

    print("\n[smoke] All phases complete. Close the window to exit.")


threading.Thread(target=_injector, daemon=True).start()

ft.run(_wrapped_main)
