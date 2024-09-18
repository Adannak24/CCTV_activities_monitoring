# from pose_module import poseDetector
# import cv2
# import time
#
# cap = cv2.VideoCapture(r"C:\Users\Public\Bluebricks\facial_recognition\Dancing.mp4")
# pTime = 0
# detector = poseDetector()
# while True:
#     success, img = cap.read()
#     # Check if the frame was successfully captured
#     if not success:
#         print("Failed to read the video or the video has ended.")
#         break
#     img = detector.findPose(img)
#     lmList = detector.findPosition(img, draw=False)
#     if len(lmList) != 0:
#         print(lmList[14])
#         cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                 (255, 0, 0), 3)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)


# import tensorflow as tf
# import cv2
# import numpy as np
#
# model = tf.saved_model.load(r'C:\Users\Public\Bluebricks\facial_recognition\movenet-tensorflow2-multipose-lightning-v1')
# movenet = model.signatures['serving_default']
#
# # Function to loop through each person detected and render
# def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
#     for person in keypoints_with_scores:
#         draw_connections(frame, person, edges, confidence_threshold)
#         draw_keypoints(frame, person, confidence_threshold)
#
#
# def draw_keypoints(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
#
#
# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }
#
#
# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
#
#         if (c1 > confidence_threshold) & (c2 > confidence_threshold):
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
#
#
# cap = cv2.VideoCapture(r'C:\Users\Public\Bluebricks\facial_recognition\Dancing.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     # Resize image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
#     input_img = tf.cast(img, dtype=tf.int32)
#
#     # Detection section
#     results = movenet(input_img)
#     keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
#
#     # Render keypoints
#     loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
#
#     cv2.imshow("Frame", frame)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# import tensorflow as tf
# import cv2
# import numpy as np
#
# model = tf.saved_model.load(r'C:\Users\Public\Bluebricks\facial_recognition\movenet-tensorflow2-multipose-lightning-v1')
# movenet = model.signatures['serving_default']
#
#
# # Function to loop through each person detected and render
# def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
#     for person in keypoints_with_scores:
#         draw_connections(frame, person, edges, confidence_threshold)
#         draw_keypoints(frame, person, confidence_threshold)
#         check_posture(frame, person, confidence_threshold)
#
#
# # Function to check if a person is seated or standing
# def check_posture(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     # Points for hips and knees (using keypoint indices from the COCO dataset)
#     left_hip, right_hip = shaped[11], shaped[12]
#     left_knee, right_knee = shaped[13], shaped[14]
#
#     # Checking confidence of the keypoints
#     if left_hip[2] > confidence_threshold and left_knee[2] > confidence_threshold and right_hip[
#         2] > confidence_threshold and right_knee[2] > confidence_threshold:
#         # Calculate distances between hip and knee for both legs
#         left_hip_knee_diff = abs(left_hip[1] - left_knee[1])
#         right_hip_knee_diff = abs(right_hip[1] - right_knee[1])
#
#         # Average hip-knee difference
#         avg_diff = (left_hip_knee_diff + right_hip_knee_diff) / 2
#         print(left_hip_knee_diff, right_hip_knee_diff, avg_diff)
#         # If the difference is small, we assume the person is seated
#         if avg_diff < 50:  # You can adjust this threshold as per your setup
#             cv2.putText(frame, 'Seated', (int(left_hip[1]), int(left_hip[0]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (255, 0, 0), 2, cv2.LINE_AA)
#         else:
#             cv2.putText(frame, 'Standing', (int(left_hip[1]), int(left_hip[0]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 255, 0), 2, cv2.LINE_AA)
#
#
# def draw_keypoints(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
#
#
# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }
#
#
# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
#
#         if (c1 > confidence_threshold) & (c2 > confidence_threshold):
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
#
#
# cap = cv2.VideoCapture(r'C:\Users\Public\Bluebricks\facial_recognition\SECURUS CCTV - 2 Megapixel IP Camera with Audio Classroom Solution (online-video-cutter.com).mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     # Resize image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
#     input_img = tf.cast(img, dtype=tf.int32)
#
#     # Detection section
#     results = movenet(input_img)
#     keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
#
#     # Render keypoints and check posture
#     loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
#
#     cv2.imshow("Frame", frame)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import tensorflow as tf
import cv2
import numpy as np

model = tf.saved_model.load(r'C:\Users\Public\Bluebricks\facial_recognition\movenet-tensorflow2-multipose-lightning-v1')
movenet = model.signatures['serving_default']

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
        check_posture(frame, person, confidence_threshold)


# Updated function to check if a person is seated or standing
def check_posture(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    # Points for hips and knees (using keypoint indices from the COCO dataset)
    left_hip, right_hip = shaped[11], shaped[12]
    left_knee, right_knee = shaped[13], shaped[14]
    left_ankle, right_ankle = shaped[15], shaped[16]

    # Checking confidence of the keypoints
    if all(kp[2] > confidence_threshold for kp in
           [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
        # Calculate vertical distances
        left_hip_ankle_diff = abs(left_hip[0] - left_ankle[0])
        right_hip_ankle_diff = abs(right_hip[0] - right_ankle[0])

        # Average hip-ankle vertical difference
        avg_diff = (left_hip_ankle_diff + right_hip_ankle_diff) / 2

        # Threshold for seated vs standing (adjust as needed)
        threshold = y * 0.1  # 20% of frame height

        if avg_diff < threshold:
            posture = 'Seated'
            color = (255, 0, 0)  # Blue for seated
        else:
            posture = 'Standing'
            color = (0, 255, 0)  # Green for standing

        # Calculate position for text (above the person's head)
        head_y = min(shaped[0][0], shaped[1][0], shaped[2][0])  # Y-coordinate of the highest point (head)
        head_x = shaped[0][1]  # X-coordinate of the nose

        # Display text above the person's head
        cv2.putText(frame, posture, (int(head_x), int(head_y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    cv2.LINE_AA)


# Rest of the code remains the same...

cap = cv2.VideoCapture(r"C:\Users\Public\Bluebricks\facial_recognition\SECURUS CCTV - 2 Megapixel IP Camera with Audio Classroom Solution (online-video-cutter.com).mp4")
while cap.isOpened():
    ret, frame = cap.read()

    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Render keypoints and check posture
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
