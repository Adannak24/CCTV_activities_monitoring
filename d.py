import cv2
from object_detection import ObjectDetection
import time
import math


class CrowdDetection:
    def __init__(self, distance_threshold=50, class_id_person=0):
        # Initialize Object Detection
        self.od = ObjectDetection()
        self.distance_threshold = distance_threshold
        self.class_id_person = class_id_person

    def process_frame(self, frame):
        """Process each frame for object detection and proximity alerts."""
        # Detect objects on the current frame
        (class_ids, scores, boxes) = self.od.detect(frame)
        center_points_cur_frame = []
        # Process only "person" objects
        for class_id, box in zip(class_ids, boxes):
            if class_id == self.class_id_person:
                (x, y, w, h) = box
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                center_points_cur_frame.append((cx, cy))

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check for proximity between detected persons
        for i, pt1 in enumerate(center_points_cur_frame):
            color = (0, 255, 0)  # Green by default
            for j, pt2 in enumerate(center_points_cur_frame):
                if i != j:
                    distance = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                    if distance < self.distance_threshold:
                        color = (0, 0, 255)  # Red if too close
                        print(f"Alert! Objects getting close: Distance = {distance:.2f}px between points {pt1} and {pt2}")
                        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Draw a line between close points
                        cv2.putText(frame, "CROWD DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        return frame, True
            # Draw the center point
            cv2.circle(frame, pt1, 5, color, -1)

        return frame, False


# # Example usage for an external frame
# if __name__ == "__main__":
#     # Initialize CrowdDetection
#     detector = CrowdDetection(distance_threshold=50)
#
#     # Example: Open a video capture from a webcam or any video stream
#     cap = cv2.VideoCapture("exam_hall (online-video-cutter.com).mp4")  # 0 for default webcam, or provide video path
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Pass the current frame to the detector
#         processed_frame = detector.process_frame(frame)
#
#         # Display the processed frame
#         cv2.imshow("Processed Frame", processed_frame)
#
#         key = cv2.waitKey(1)
#         if key == 27:  # Esc key to break
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
