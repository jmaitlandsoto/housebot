"""housebot — motion-gated object detection on a Tapo RTSP stream."""
import os
import shutil
import sys
import time
from pathlib import Path

import av
import cv2
import yaml
import numpy as np
import pygame
from dotenv import load_dotenv
from ultralytics import YOLO

# ---------------------------------------------------------------- config

load_dotenv()

def _require_env(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"missing {name} — copy .env.example to .env and fill in your camera's details")
    return val

CAMERA_USER = _require_env("CAMERA_USER")
CAMERA_PASS = _require_env("CAMERA_PASS")
CAMERA_HOST = _require_env("CAMERA_HOST")
RTSP_URL    = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_HOST}:554/stream2"
OV_DIR     = Path("yolo26n_int8_openvino_model")
IMGSZ      = 640
KEEP       = {0: "person", 2: "car", 5: "bus", 7: "truck"}  # COCO classes to keep

MOTION_PX  = 100000   # changed pixels that count as motion
LINGER     = 30     # keep inferring this many frames after motion stops
WARMUP     = 40     # frames MOG2 needs before its background is usable
SHOW       = True   # preview window; set False to run headless
SHOW_W     = 960    # preview width — smaller costs less to draw

# ---------------------------------------------------------------- model

def load_model():
    # If a cached export exists, verify its imgsz matches the current setting.
    meta = OV_DIR / "metadata.yaml"
    if meta.exists():
        stored_imgsz = yaml.safe_load(meta.read_text()).get("imgsz", [])
        if stored_imgsz != [IMGSZ, IMGSZ]:
            print(f"[housebot] IMGSZ changed (stored={stored_imgsz}, current={IMGSZ}); clearing cache...")
            shutil.rmtree(OV_DIR)

    # NNCF quantization can fail on some torch/numpy combinations;
    # FP32 OpenVINO is still much faster than PyTorch, so fall back.
    if not OV_DIR.exists():
        print("[housebot] exporting OpenVINO model (one time)...")
        try:
            YOLO("yolo26n.pt").export(
                format="openvino", imgsz=IMGSZ,
                quantize="int8", data="coco128.yaml",
            )
        except Exception as exc:
            print(f"[housebot] INT8 export failed ({exc}); using FP32")
            YOLO("yolo26n.pt").export(format="openvino", imgsz=IMGSZ)
            return YOLO(Path("yolo26n_openvino_model"))
    return YOLO(OV_DIR)


def pick_device(model):
    """Try the iGPU, fall back to CPU if OpenCL isn't usable."""
    probe = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for dev in ("intel:gpu", "cpu"):
        try:
            model.predict(probe, device=dev, imgsz=IMGSZ, verbose=False)
            print(f"[housebot] inference on {dev}")
            return dev
        except Exception as exc:
            print(f"[housebot] {dev} unavailable: {type(exc).__name__}")
    return "cpu"

# ---------------------------------------------------------------- stream

def frames(url):
    """Yield BGR frames forever, reconnecting on failure."""
    while True:
        container = None
        try:
            container = av.open(url, options={
                "rtsp_transport": "tcp",
                "max_delay": "500000",
                "timeout": "5000000",
            })
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            print("[housebot] stream open")
            for packet in container.demux(stream):
                for frame in packet.decode():
                    yield frame.to_ndarray(format="bgr24")
        except Exception as exc:
            print(f"[housebot] stream error: {exc}; retrying in 3s")
            time.sleep(3)
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

# ---------------------------------------------------------------- main

def main():
    model  = load_model()
    device = pick_device(model)
    bg     = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    live   = {}    # track id -> [label, first_seen, last_seen]
    idle   = LINGER
    seen   = 0

    if SHOW:
        pygame.init()
        pygame.display.set_caption("housebot")
        screen = None   # sized lazily on first frame

    for frame in frames(RTSP_URL):
        seen += 1
        moved  = cv2.countNonZero(bg.apply(frame))
        now    = time.time()
        canvas = frame          # what the preview shows this pass
        state  = "WARMUP"

        if seen >= WARMUP:       # MOG2 calls everything motion at first
            # if moved >= MOTION_PX:
            #     idle = 0
            # else:
            #     idle += 1

            # if idle <= LINGER:   # quiet scene: skip the network entirely
                state = "ACTIVE"
                results = model.track(
                    frame, persist=True, device=device, imgsz=IMGSZ,
                    classes=list(KEEP), tracker="bytetrack.yaml", verbose=False,
                )

                for r in results:
                    canvas = r.plot()          # boxes, labels, track ids
                    for box in r.boxes:
                        if box.id is None:
                            continue
                        tid   = int(box.id)
                        label = KEEP[int(box.cls)]
                        conf  = float(box.conf)
                        if tid not in live:
                            live[tid] = [label, now, now]
                            print(f"[+] {label} #{tid} conf={conf:.2f}")
                        else:
                            live[tid][2] = now

                for tid in [t for t, v in live.items() if now - v[2] > 3.0]:
                    label, first, last = live.pop(tid)
                    print(f"[-] {label} #{tid} gone after {last - first:.1f}s")
            # else:
            #     state = "IDLE"

        if SHOW:
            h, w = canvas.shape[:2]
            if w > SHOW_W:
                canvas = cv2.resize(canvas, (SHOW_W, int(h * SHOW_W / w)))
            colour = (0, 200, 0) if state == "ACTIVE" else (160, 160, 160)
            cv2.putText(canvas, f"{state}  motion={moved}  tracks={len(live)}",
                        (12-2, 48+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA,  False)
            cv2.putText(canvas, f"{state}  motion={moved}  tracks={len(live)}",
                        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 1, cv2.LINE_AA, False)

            # convert BGR numpy array → pygame surface and display
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if screen is None or screen.get_size() != (w, h):
                screen = pygame.display.set_mode((w, h))
            pygame.surfarray.blit_array(screen, rgb.transpose(1, 0, 2))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return

    if SHOW:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[housebot] stopped")
    finally:
        if SHOW:
            pygame.quit()
