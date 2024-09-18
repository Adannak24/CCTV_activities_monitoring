import cv2
from object_detection import ObjectDetection
import time
import math

# Initialize Object Detection
od = ObjectDetection()

# Assuming the class ID for "person" is 1 (this is common, but adjust based on your model)
class_id_person = 0

cap = cv2.VideoCapture("combined_checks.mp4")

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = int(1000 / fps)  # Time per frame in milliseconds

# Initialize count
count = 0

# Threshold distance to trigger an alert
distance_threshold = 50  # Adjust based on your needs

while True:
    start_time = time.time()  # Start time measurement

    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    print(class_ids)
    for class_id, box in zip(class_ids, boxes):
        if class_id == class_id_person:  # Only process "person" objects
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
            if i != j:  # Ensure not comparing the same point
                distance = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if distance < distance_threshold:
                    color = (0, 0, 255)  # Change to red if too close
                    print(f"Alert! Objects getting close: Distance = {distance:.2f}px between points {pt1} and {pt2}")
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Draw a line between the two close points
                    cv2.putText(frame, "CROWD DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw the center point
        cv2.circle(frame, pt1, 5, color, -1)

    # Measure processing time and adjust wait time
    end_time = time.time()
    processing_time = int((end_time - start_time) * 1000)
    wait_time = max(1, frame_time - processing_time)

    print(f"Frame: {count}, Processing Time: {processing_time}ms, Wait Time: {wait_time}ms")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(frame_time)
    if key == 27:  # Esc key to break
        break

cap.release()
cv2.destroyAllWindows()
