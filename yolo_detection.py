from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Dict, Tuple
from torch.nn.modules.container import Sequential

torch.serialization.add_safe_globals([
    torch.nn.Module,
    Sequential,
    __import__('ultralytics.nn.tasks').nn.tasks.DetectionModel
])

class FoodDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.4

    def detect_and_visualize(self, image_bytes: bytes) -> Tuple[Dict[str, float], bytes]:
        """Обрабатывает изображение в формате bytes"""
        try:
            # Конвертируем bytes в numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Не удалось декодировать изображение")
            
            # Детекция
            results = self.model(img)
            detected = {}
            visualized = img.copy()
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf >= self.confidence_threshold:
                        label = result.names[int(box.cls)]
                        detected[label] = conf
                        
                        # Рисуем bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(visualized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(visualized, f"{label} {conf:.2f}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
            
            # Конвертируем обратно в bytes
            _, img_encoded = cv2.imencode('.jpg', visualized)
            return detected, img_encoded.tobytes()
            
        except Exception as e:
            print(f"Ошибка детекции: {str(e)}")
            raise