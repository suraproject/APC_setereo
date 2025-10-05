"""Simple wrapper for running YOLOv8 on a frame and returning boxes of people."""
from ultralytics import YOLO
import numpy as np

class YoloPersonDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        # returns list of (x1,y1,x2,y2,conf)
        results = self.model(frame, conf=self.conf)
        boxes = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0]) if hasattr(b, 'conf') else 1.0
                boxes.append((x1,y1,x2,y2,conf))
        return boxes

if __name__ == '__main__':
    import cv2
    det = YoloPersonDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, f = cap.read()
        if not ret: break
        boxes = det.detect(f)
        for (x1,y1,x2,y2,c) in boxes:
            cv2.rectangle(f, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow('yolo test', f)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()
