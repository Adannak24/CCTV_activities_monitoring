import cv2
import math
from object_detection import ObjectDetection


class ObjectTracker:
    def __init__(self):
        # Initialize Object Detection here
        self.od = ObjectDetection()
        self.class_id_person = 0  # Adjust based on your model

    def process_frame(self, frame):
        center_points_cur_frame = []
        (class_ids, scores, boxes) = self.od.detect(frame)
        for class_id, box in zip(class_ids, boxes):
            if class_id == self.class_id_person:
                (x, y, w, h) = box
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                center_points_cur_frame.append((cx, cy))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i, pt1 in enumerate(center_points_cur_frame):
            color = (0, 255, 0)
            for j, pt2 in enumerate(center_points_cur_frame):
                if i != j:
                    distance = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                    if distance < 50:
                        color = (0, 0, 255)
                        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
                        cv2.putText(frame, "CROWD DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(frame, pt1, 5, color, -1)
        return frame
