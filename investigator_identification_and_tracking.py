# import cv2
# import mediapipe as mp
#
# # Initialize the MediaPipe Pose Estimation model
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
#
#
# # Draw function to draw bounding boxes with different colors
# def draw_bounding_box(frame, bbox, color):
#     (x, y, w, h) = [int(v) for v in bbox]
#     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#
#
# # Process video frame
# cap = cv2.VideoCapture('videoplayback.mp4')  # You can replace this with your live feed
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Run pose estimation
#     results = pose.process(frame_rgb)
#
#     # Get image dimensions for bounding box calculations
#     height, width, _ = frame.shape
#
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#
#         # Get key points (head and knees)
#         head_y = landmarks[mp_pose.PoseLandmark.NOSE].y * height
#         left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * height
#         right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * height
#
#         # Get bounding box for the person (using shoulder and knee points)
#         left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width
#         right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width
#         knee_y = min(left_knee_y, right_knee_y)
#
#         # Define the bounding box area
#         bbox = [left_shoulder_x, head_y, right_shoulder_x - left_shoulder_x, knee_y - head_y]
#
#         # Determine if the person is standing or sitting based on head-to-knee distance
#         if head_y < left_knee_y and head_y < right_knee_y:  # Standing
#             draw_bounding_box(frame, bbox, (0, 0, 255))  # Red for standing
#             cv2.putText(frame, "Standing", (int(left_shoulder_x), int(head_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 0, 255), 2, cv2.LINE_AA)
#         else:  # Sitting
#             draw_bounding_box(frame, bbox, (0, 255, 0))  # Green for sitting
#             cv2.putText(frame, "Sitting", (int(left_shoulder_x), int(head_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 255, 0), 2, cv2.LINE_AA)
#
#     # Display the frame
#     cv2.imshow('Pose Estimation with Bounding Boxes', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# from object_detection import ObjectDetection
# import time
# import math
#
# # Initialize Object Detection
# od = ObjectDetection()
#
# # Assuming the class ID for "person" is 1 (adjust based on your model)
# class_id_person = 0
#
# cap = cv2.VideoCapture(r"C:\Users\Public\Bluebricks\facial_recognition\exam_hall (online-video-cutter.com).mp4")
#
# # Get the frame rate of the video
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_time = int(1000 / fps)  # Time per frame in milliseconds
#
# # Initialize count
# count = 0
#
# # Store center point history to track movement
# center_point_history = {}
# movement_threshold = 50  # Pixels to consider significant movement
# distance_threshold = 50  # Distance between bounding boxes to trigger proximity alert
#
# # Set exam hall boundaries (e.g., based on pixel coordinates)
# exam_hall_boundaries = (50, 50, 600, 400)  # Example (xmin, ymin, xmax, ymax)
#
#
# # Helper function to calculate movement
# def calculate_movement(old_points, new_point):
#     if old_points:
#         prev_point = old_points[-1]  # Get the last recorded point
#         movement = math.sqrt((prev_point[0] - new_point[0]) ** 2 + (prev_point[1] - new_point[1]) ** 2)
#         return movement
#     return 0
#
#
# while True:
#     start_time = time.time()
#
#     ret, frame = cap.read()
#     count += 1
#     if not ret:
#         break
#
#     center_points_cur_frame = []
#     bounding_boxes_cur_frame = []
#
#     # Detect objects in the current frame
#     (class_ids, scores, boxes) = od.detect(frame)
#     for class_id, score, box in zip(class_ids, scores, boxes):
#         if class_id == class_id_person and score > 0.5:  # Only process "person" objects with confidence > 0.5
#             (x, y, w, h) = box
#             cx = int((x + x + w) / 2)
#             cy = int((y + y + h) / 2)
#             center_points_cur_frame.append((cx, cy))
#             bounding_boxes_cur_frame.append((x, y, w, h))
#
#             # Draw green bounding box for students
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Find the largest bounding box (assumed to be the investigator)
#     largest_box = None
#     largest_box_size = 0
#     largest_center_point = None
#     for i, (box, center) in enumerate(zip(bounding_boxes_cur_frame, center_points_cur_frame)):
#         box_size = box[2] * box[3]  # Width * Height
#         if box_size > largest_box_size:
#             largest_box_size = box_size
#             largest_box = box
#             largest_center_point = center
#
#     # Track movement of the investigator (largest bounding box)
#     if largest_center_point:
#         movement = calculate_movement(center_point_history.get('investigator', []), largest_center_point)
#
#         # Update history
#         if 'investigator' in center_point_history:
#             center_point_history['investigator'].append(largest_center_point)
#         else:
#             center_point_history['investigator'] = [largest_center_point]
#
#         # Draw investigator bounding box in red
#         (x, y, w, h) = largest_box
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for the investigator
#         cv2.putText(frame, "Investigator", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#         # Draw investigator center point
#         cv2.circle(frame, largest_center_point, 5, (255, 0, 0), -1)  # Blue for the investigator's center point
#
#         # Check if investigator moves significantly
#         if movement > movement_threshold:
#             print(f"Investigator is moving: {movement:.2f}px")
#         else:
#             print(f"Investigator is stationary: {movement:.2f}px")
#
#         # Check if the investigator goes outside exam hall boundaries
#         xmin, ymin, xmax, ymax = exam_hall_boundaries
#         # if not (xmin <= largest_center_point[0] <= xmax and ymin <= largest_center_point[1] <= ymax):
#         #     print("Alert! Investigator has left the exam hall!")
#
#     # Measure processing time and adjust wait time
#     end_time = time.time()
#     processing_time = int((end_time - start_time) * 1000)
#     wait_time = max(1, frame_time - processing_time)
#
#     print(f"Frame: {count}, Processing Time: {processing_time}ms, Wait Time: {wait_time}ms")
#
#     cv2.imshow("Exam Hall", frame)
#
#     key = cv2.waitKey(wait_time)
#     if key == 27:  # Esc key to break
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Helper function to calculate movement
def calculate_movement(old_points, new_point):
    if old_points:
        prev_point = old_points[-1]  # Get the last recorded point
        movement = math.sqrt((prev_point[0] - new_point[0]) ** 2 + (prev_point[1] - new_point[1]) ** 2)
        return movement
    return 0



# Set initial frame size boundaries
def set_exam_hall_boundaries(frame):
    global exam_hall_boundaries
    height, width = frame.shape[:2]
    xmin, ymin = 0, 0
    xmax, ymax = width, height
    exam_hall_boundaries = (xmin, ymin, xmax, ymax)

import cv2
from object_detection import ObjectDetection
import time
import math

# Initialize Object Detection
od = ObjectDetection()

# Assuming the class ID for "person" is 1 (adjust based on your model)
class_id_person = 0

cap = cv2.VideoCapture("SECURUS CCTV - 2 Megapixel IP Camera with Audio Classroom Solution (online-video-cutter.com).mp4")

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = int(1000 / fps)  # Time per frame in milliseconds

# Initialize variables
count = 0
detected_investigator = None
investigator_detected = False
frames_to_check = 50  # Number of frames to observe before locking onto the investigator
frame_history = []
movement_threshold = 50  # Pixels to consider significant movement

# Set exam hall boundaries (e.g., based on pixel coordinates)
exam_hall_boundaries = None  # Initialize boundaries
# Initialize history to track center points
center_points_history = []
# Helper function to calculate movement
def calculate_movement(old_points, new_point):
    if old_points:
        prev_point = old_points[-1]  # Get the last recorded point
        movement = math.sqrt((prev_point[0] - new_point[0]) ** 2 + (prev_point[1] - new_point[1]) ** 2)
        return movement
    return 0

while True:
    start_time = time.time()

    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    if count == 1:
        # Set the exam hall boundaries based on the first frame
        set_exam_hall_boundaries(frame)

    center_points_cur_frame = []
    bounding_boxes_cur_frame = []

    # Detect objects in the current frame
    (class_ids, scores, boxes) = od.detect(frame)
    for class_id, score, box in zip(class_ids, scores, boxes):
        if class_id == class_id_person and score > 0.5:  # Only process "person" objects with confidence > 0.5
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
            bounding_boxes_cur_frame.append((x, y, w, h))

            # Draw green bounding box for detected persons
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # If investigator has not been identified yet, check the first 20 frames
    if not investigator_detected and count <= frames_to_check:
        largest_box = None
        largest_box_size = 0
        largest_center_point = None

        # Find the largest bounding box for the current frame
        for i, (box, center) in enumerate(zip(bounding_boxes_cur_frame, center_points_cur_frame)):
            box_size = box[2] * box[3]  # Width * Height
            if box_size > largest_box_size:
                largest_box_size = box_size
                largest_box = box
                largest_center_point = center

        # Store the largest box and center point for this frame
        frame_history.append((largest_box, largest_center_point))

        # Once 20 frames have been checked, lock onto the most consistent largest box
        if count == frames_to_check:
            # Find the most consistent person over 20 frames
            consistent_box = max(frame_history, key=lambda b: b[0][2] * b[0][3])[0]
            detected_investigator = consistent_box
            investigator_detected = True
            print(f"Investigator identified after {frames_to_check} frames.")

    # If investigator is detected, only track them and ignore others
    if investigator_detected:
        (x, y, w, h) = detected_investigator
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        # Draw investigator bounding box in red
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for the investigator
        cv2.putText(frame, "Investigator", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw the center point
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Check if the investigator goes outside exam hall boundaries
        xmin, ymin, xmax, ymax = exam_hall_boundaries
        if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
            print("Alert! Investigator has left the exam hall!")

        # Track the movement of the center point
        movement = calculate_movement(center_points_history, (cx, cy))
        if movement > movement_threshold:
            print(f"Investigator moving: {movement:.2f} pixels")

        # Update center points history
        center_points_history.append((cx, cy))

        # Ensure the bounding box follows the investigator's movement
        # You may consider re-checking the current frame for the bounding box closest to the previously detected center
        closest_distance = float('inf')
        for i, (box, center) in enumerate(zip(bounding_boxes_cur_frame, center_points_cur_frame)):
            distance = math.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                detected_investigator = box

    # Measure processing time and adjust wait time
    end_time = time.time()
    processing_time = int((end_time - start_time) * 1000)
    wait_time = max(1, frame_time - processing_time)

    print(f"Frame: {count}, Processing Time: {processing_time}ms, Wait Time: {wait_time}ms")

    cv2.imshow("Exam Hall", frame)

    key = cv2.waitKey(wait_time)
    if key == 27:  # Esc key to break
        break

cap.release()
cv2.destroyAllWindows()

