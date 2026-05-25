import logging
import os
import threading
from datetime import datetime

import flet as ft

import audio_loop

os.makedirs("logs", exist_ok=True)
_session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

_file_handler = logging.FileHandler(f"logs/{_session_id}.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_fmt)

logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(_file_handler)
logging.root.addHandler(_console_handler)

log = logging.getLogger("main")


def main(page: ft.Page):
    page.title = "Yammer"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.bgcolor = "#1a1a1a"
    page.padding = 0
    page.window.width = 960
    page.window.height = 640
    page.window.min_width = 700
    page.window.min_height = 480

    status = ft.Text("Starting session...", size=16)
    hint = ft.Text("", size=13, color=ft.Colors.GREY_600)
    mic_level = ft.ProgressBar(value=0, width=200, color=ft.Colors.GREEN, bgcolor=ft.Colors.GREY_300, visible=False)
    mic_label = ft.Text("Hold to Speak", size=16, color=ft.Colors.WHITE)
    mic_button = ft.Container(
        content=mic_label,
        bgcolor=ft.Colors.BLUE,
        border_radius=8,
        padding=ft.Padding(left=40, right=40, top=20, bottom=20),
        disabled=True,
        opacity=0.4,
    )

    # ── Chat panel ──────────────────────────────────────────────────────────

    chat_list = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
        padding=ft.Padding(left=12, right=12, top=8, bottom=8),
    )

    def add_message(role: str, text: str):
        is_user = role == "user"
        bubble = ft.Container(
            content=ft.Text(text, size=13, color=ft.Colors.WHITE, selectable=True),
            bgcolor=ft.Colors.BLUE_700 if is_user else "#2d4a2d",
            border_radius=12,
            padding=ft.Padding(left=12, right=12, top=8, bottom=8),
            width=280,
        )
        label = ft.Text(
            "You" if is_user else "Yammer",
            size=11,
            color=ft.Colors.GREY_500,
        )
        col = ft.Column(
            [label, bubble],
            spacing=2,
            horizontal_alignment=ft.CrossAxisAlignment.END if is_user else ft.CrossAxisAlignment.START,
        )
        chat_list.controls.append(
            ft.Row(
                [col],
                alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START,
            )
        )
        try:
            page.update()
        except Exception:
            pass

    # ── Status / button callbacks ────────────────────────────────────────────

    def on_status(state: str):
        log.info("Status -> %s", state)
        if state == "connecting":
            status.value = "Connecting..."
            mic_button.disabled = True
            mic_button.opacity = 0.4
            mic_button.bgcolor = ft.Colors.BLUE
        elif state == "ready":
            status.value = "Press and hold to speak"
            mic_button.disabled = False
            mic_button.opacity = 1.0
            mic_button.bgcolor = ft.Colors.GREEN
            mic_button.animate_opacity = ft.Animation(500, ft.AnimationCurve.EASE_IN_OUT)
        elif state == "speaking":
            status.value = "Yammer is speaking..."
            mic_button.disabled = True
            mic_button.opacity = 0.4
            mic_button.bgcolor = ft.Colors.BLUE
            mic_button.animate_opacity = None
        elif state == "listening":
            status.value = "Listening..."
            mic_button.bgcolor = ft.Colors.RED
            mic_button.opacity = 1.0
            mic_level.visible = True
            mic_level.value = 0
        elif state == "processing":
            status.value = "Processing..."
            mic_button.bgcolor = ft.Colors.ORANGE
            mic_button.disabled = True
            mic_button.opacity = 0.6
            mic_button.animate_opacity = None
            mic_level.visible = False
            mic_level.value = 0
        page.update()

    def start_listening(e):
        log.info("Tap down — starting recording")
        on_status("listening")
        threading.Thread(target=_record, daemon=True).start()

    def stop_listening(e):
        log.info("Tap up — stopping recording")
        audio_loop.stop_recording()
        on_status("processing")

    _current_level = [0.0]

    def _on_mic_level(level: float):
        _current_level[0] = level

    def _level_poll_thread():  # pragma: no cover
        import time
        while True:
            time.sleep(0.1)
            if mic_level.visible:
                mic_level.value = _current_level[0]
                try:
                    page.update()
                except Exception:
                    pass

    threading.Thread(target=_level_poll_thread, daemon=True).start()

    def _record():
        log.info("Recording thread started")
        audio_loop.record_and_stream(on_level=_on_mic_level)
        log.info("Recording thread done")

    mic_gesture = ft.GestureDetector(
        content=mic_button,
        on_tap_down=start_listening,
        on_tap_up=stop_listening,
    )

    # ── Layout ───────────────────────────────────────────────────────────────

    left_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("Yammer", size=32, weight=ft.FontWeight.BOLD),
                ft.Text("French Language Tutor", size=14, color=ft.Colors.GREY_600),
                ft.Divider(height=40, color=ft.Colors.TRANSPARENT),
                mic_gesture,
                ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                status,
                mic_level,
                hint,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        ),
        width=300,
        padding=ft.Padding(left=24, right=24, top=24, bottom=24),
    )

    right_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("Conversation", size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_500),
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                chat_list,
            ],
            expand=True,
            spacing=8,
        ),
        expand=True,
        bgcolor="#111111",
        padding=ft.Padding(left=16, right=16, top=16, bottom=16),
        border=ft.Border(left=ft.BorderSide(1, ft.Colors.GREY_800)),
    )

    page.add(
        ft.Row(
            [left_panel, right_panel],
            expand=True,
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
    )

    log.info("UI ready — starting background music and session")
    audio_loop.set_message_callback(add_message)
    audio_loop.play_mp3_background(r"C:\Users\karst\Downloads\happy-day-paris.mp3", volume=0.2)
    audio_loop.start_session(on_status)


ft.run(main)  # pragma: no cover
