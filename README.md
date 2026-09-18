# housebot

Motion-gated object detection on a Tapo RTSP camera stream.

## Setup

1. Clone the repo and create a virtualenv:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your camera's RTSP username, password, and host/IP:

   ```
   cp .env.example .env
   ```

   > [!NOTE]
   > In the Tapo App go to Device Settings > Advanced Settings > Camera Account > Toggle On Camera Account to setup a username and password

3. Run it:
   ```
   python index.py
   ```

Set `SHOW = False` in `index.py` to run headless (no preview window).
