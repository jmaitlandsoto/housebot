import os
import sys
import time
from pathlib import Path

import av
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # .env lives at repo root, not tests/

def _require_env(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"missing {name} — copy .env.example to .env and fill in your camera's details")
    return val

CAMERA_USER = _require_env("CAMERA_USER")
CAMERA_PASS = _require_env("CAMERA_PASS")
CAMERA_HOST = _require_env("CAMERA_HOST")
URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_HOST}:554/stream1"

container = av.open(URL, options={
    "rtsp_transport": "tcp",
    "max_delay": "500000",
})
stream = container.streams.video[0]
stream.thread_type = "AUTO"

n, t0 = 0, time.time()
for packet in container.demux(stream):
    for _ in packet.decode():
        n += 1
        if n >= 300:
            break
    if n >= 300:
        break

print(f"{n / (time.time() - t0):.1f} fps decode-only")
