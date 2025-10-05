# APC_setereo — Stereo Door People Counter

Stereo-camera people counter (two IP cameras mounted above a door).
Modules: calibration, depth estimation, YOLO detection, head-tracking, and counting.

## Quick start
1. Copy `.env.example` to `.env` locally and fill camera RTSP URLs.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
