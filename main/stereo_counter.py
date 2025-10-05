"""Main integration file. Run this to start RTSP captures from two cameras, perform detection, tracking, and counting.

Usage:
    python stereo_counter.py --cam1 <rtsp1> --cam2 <rtsp2>
"""
import cv2
import numpy as np
import threading
import argparse
import time
from detection.yolo_detect import YoloPersonDetector
from detection.head_tracker import CentroidTracker, check_crossing
from depth_estimation.compute_depth_map import compute_disparity
from main.utils import get_env_var
import signal
import sys

class CameraThread(threading.Thread):
    def __init__(self, src, name='cam'):
        super().__init__()
        self.src = src
        self.name = name
        self.capture = cv2.VideoCapture(self.src)
        self.frame = None
        self.stopped = False
        # try to set resolution (may be ignored by RTSP device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        while not self.stopped:
            ret, f = self.capture.read()
            if not ret:
                time.sleep(0.05)
                continue
            self.frame = f

    def stop(self):
        self.stopped = True
        try:
            self.capture.release()
        except Exception:
            pass

def graceful_exit(signum, frame):
    print('\nReceived exit signal, terminating...')
    sys.exit(0)

def is_likely_head(bbox, depth_map=None, min_head_h=30, max_head_h=250):
    # Basic heuristic: bounding box height and optional depth median check
    x1,y1,x2,y2 = bbox
    h = y2 - y1
    if h < min_head_h or h > max_head_h:
        return False
    if depth_map is not None:
        H, W = depth_map.shape
        x1c = max(0, min(W-1, x1)); x2c = max(0, min(W-1, x2))
        y1c = max(0, min(H-1, y1)); y2c = max(0, min(H-1, y2))
        patch = depth_map[y1c:y2c, x1c:x2c]
        if patch.size == 0:
            return False
        median_depth = np.median(patch)
        if np.isnan(median_depth) or median_depth <= 0 or median_depth > 2000:
            return False
    return True

def main(cam1_url, cam2_url, model_path='yolov8n.pt'):
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    cam1 = CameraThread(cam1_url, 'cam1')
    cam2 = CameraThread(cam2_url, 'cam2')
    cam1.start(); cam2.start()

    detector = YoloPersonDetector(model_path=model_path, conf=0.35)
    tracker1 = CentroidTracker(max_disappeared=40, max_distance=80)
    tracker2 = CentroidTracker(max_disappeared=40, max_distance=80)

    people_in = 0
    people_out = 0

    window_name = 'Stereo People Counter'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            f1 = cam1.frame.copy() if cam1.frame is not None else None
            f2 = cam2.frame.copy() if cam2.frame is not None else None

            if f1 is None and f2 is None:
                time.sleep(0.05)
                continue
            if f1 is None:
                f1 = 255 * np.ones_like(f2)
            if f2 is None:
                f2 = 255 * np.ones_like(f1)

            # compute depth (small resize for speed)
            smallL = cv2.resize(f1, (640, 360))
            smallR = cv2.resize(f2, (640, 360))
            disp, _ = compute_disparity(smallL, smallR)
            # resize disparity back to original if needed
            depth_map = cv2.resize(disp, (f1.shape[1], f1.shape[0]))

            # per-camera detection and tracking
            for f, tracker in zip([f1, f2], [tracker1, tracker2]):
                boxes = detector.detect(f)  # list of (x1,y1,x2,y2,conf)
                # filter and produce rects
                rects = []
                for (x1,y1,x2,y2,c) in boxes:
                    if not is_likely_head((x1,y1,x2,y2), depth_map=None):
                        continue
                    rects.append((x1,y1,x2,y2))

                objects, history = tracker.update(rects)

                # draw
                h = f.shape[0]
                band_h = int(h * 0.12)
                band_center = h // 2
                line_top = band_center - band_h//2
                line_bottom = band_center + band_h//2
                cv2.rectangle(f, (0,line_top), (f.shape[1], line_bottom), (255,0,0), 2)

                # iterate objects
                for oid, centroid in objects.items():
                    hist = history[oid]
                    cx, cy = centroid
                    crossing = check_crossing(hist, line_top, line_bottom)
                    if crossing == 'in':
                        people_in += 1
                        history[oid] = [hist[-1]]
                    elif crossing == 'out':
                        people_out += 1
                        history[oid] = [hist[-1]]
                    cv2.circle(f, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(f, f"ID:{oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            combined = cv2.hconcat([f1, f2])
            cv2.putText(combined, f"In: {people_in}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(combined, f"Out: {people_out}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow(window_name, combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam1.stop(); cam2.stop()
        cam1.join(); cam2.join()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam1', required=False, default=get_env_var('CAM1') or 0)
    parser.add_argument('--cam2', required=False, default=get_env_var('CAM2') or 1)
    parser.add_argument('--model', default='yolov8n.pt')
    args = parser.parse_args()
    main(args.cam1, args.cam2, model_path=args.model)
