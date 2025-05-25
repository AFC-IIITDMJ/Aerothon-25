from ultralytics import YOLO
import cv2
from typing import Optional, Tuple

class YOLODetector:
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def detect(self, image: cv2.Mat) -> Optional[Tuple[int, int, int, int]]:
        results = self.model(image)[0]
        if len(results.boxes) == 0:
            return None
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            if conf >= 0.7:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label_text = f'{label} {conf:.2f}'
                cv2.putText(image, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 255, 25), 2)
                return x1, y1, x2, y2
        return None