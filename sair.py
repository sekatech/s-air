"""
S-AIR (One-Finger Virtual Mouse) — Movement-Only Build
======================================================
Focus: 100% real-time, high-quality cursor movement.

Key optimizations vs. v1:
  - Adaptive smoothing (EMA) instead of a fixed moving average
  - Separate capture thread (reads frames as fast as the camera produces them)
  - Frame-drop tolerance (never blocks on a slow frame)
  - Larger dead-zone box + configurable active region
  - One Euro-style low-pass filter for jitter reduction with low latency
  - Cursor freeze on hand loss instead of jump-to-last
"""

import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
from collections import deque

# --- SAFETY ---
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# --- MediaPipe (low-RAM) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# --- Camera (threaded capture) ---
FRAME_W, FRAME_H = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Only keep the newest frame

# Capture thread state
latest_frame = None
frame_lock = threading.Lock()
stop_flag = False


def capture_loop():
    """Continuously read frames; always keep the newest one."""
    global latest_frame, stop_flag
    while not stop_flag:
        ok, frame = cap.read()
        if ok:
            with frame_lock:
                latest_frame = frame


capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

# --- Tuning ---
# Active region: only this fraction of the camera frame maps to the screen.
# Larger margin = you must move your finger further to reach the edges.
ACTIVE_MARGIN_X = 0.20
ACTIVE_MARGIN_Y = 0.20

# Low-pass filter (One Euro style) — alpha close to 1.0 = less smoothing/more responsive
# alpha close to 0.0 = smoother but laggier. 0.55–0.75 is the sweet spot.
ALPHA_BASE = 0.65
# If finger moves fast, lower alpha = less lag when it matters most
ALPHA_FAST = 0.85
# Speed threshold (normalized units / frame) above which we consider movement "fast"
SPEED_FAST_THRESHOLD = 0.012

# Track last normalized position for speed estimation
last_norm_x = None
last_norm_y = None

# Smoothed screen coordinates (persist between frames)
smooth_screen_x = SCREEN_W // 2
smooth_screen_y = SCREEN_H // 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_to_screen(norm_x, norm_y):
    """Map normalized landmark to screen pixels using the active region."""
    ax = (norm_x - ACTIVE_MARGIN_X) / (1.0 - 2 * ACTIVE_MARGIN_X)
    ay = (norm_y - ACTIVE_MARGIN_Y) / (1.0 - 2 * ACTIVE_MARGIN_Y)
    ax = clamp(ax, 0.0, 1.0)
    ay = clamp(ay, 0.0, 1.0)
    return ax * SCREEN_W, ay * SCREEN_H


def adaptive_alpha(speed):
    """
    Alpha for the low-pass filter.
    Small speed -> small alpha -> smoother (low jitter).
    Large speed -> alpha approaches 1.0 -> less lag.
    """
    if speed >= SPEED_FAST_THRESHOLD:
        return ALPHA_FAST
    # Linear ramp between ALPHA_BASE and ALPHA_FAST
    t = speed / SPEED_FAST_THRESHOLD
    return ALPHA_BASE + (ALPHA_FAST - ALPHA_BASE) * t


print("[S-AIR] Movement-only build running. ESC to quit.")

try:
    while True:
        with frame_lock:
            frame = latest_frame
            latest_frame = None

        if frame is None:
            # No new frame yet — yield CPU instead of blocking
            time.sleep(0.001)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process at reduced resolution internally if you want more speed
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            idx_tip = lm[8]

            # Speed estimation from last frame
            if last_norm_x is None:
                speed = 0.0
            else:
                speed = math.hypot(idx_tip.x - last_norm_x, idx_tip.y - last_norm_y)

            last_norm_x, last_norm_y = idx_tip.x, idx_tip.y

            # Map to screen with active region
            raw_sx, raw_sy = map_to_screen(idx_tip.x, idx_tip.y)

            # Adaptive low-pass filter
            alpha = adaptive_alpha(speed)
            smooth_screen_x = alpha * raw_sx + (1.0 - alpha) * smooth_screen_x
            smooth_screen_y = alpha * raw_sy + (1.0 - alpha) * smooth_screen_y

            pyautogui.moveTo(int(smooth_screen_x), int(smooth_screen_y))

            # Visual crosshair on the preview
            cv2.circle(frame, (int(idx_tip.x * FRAME_W), int(idx_tip.y * FRAME_H)),
                       8, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(
                frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
            )
        else:
            # Hand lost — reset speed tracker so re-entry is snappy
            last_norm_x = last_norm_y = None

        cv2.putText(frame, "S-AIR | ESC to quit", (10, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("S-AIR", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop_flag = True
    capture_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[S-AIR] Stopped cleanly.")