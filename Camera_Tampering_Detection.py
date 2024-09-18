# import cv2
# import numpy as np
#
# def detect_blur(image, threshold=100):
#     gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
#     # print(f"Blur Metric: {laplacian_var}")
#     return laplacian_var < threshold
#
# def detect_blank(image, dark_threshold=40, bright_threshold=215):
#     gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mean_pixel_value = np.mean(gray_frame)
#     # print(f"Blank Frame Metric: {mean_pixel_value}")
#     return mean_pixel_value < dark_threshold or mean_pixel_value > bright_threshold
#
# def detect_flash(prev_frame, current_frame, threshold=50):
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#     brightness_diff = np.mean(curr_gray) - np.mean(prev_gray)
#     print(f"Flash Detection Metric: {brightness_diff}")
#     return brightness_diff > threshold
#
# cap = cv2.VideoCapture("WhatsApp Video 2024-09-04 at 3.22.34 PM.mp4")
#
# ret, prev_frame = cap.read()  # Initialize previous frame for flash detection
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Camera tampering detected: No frame captured!")
#         break
#     #
#     # # Detect Blur
#     # if detect_blur(frame):
#     #     print("Camera tampering detected: Blur!")
#     #
#     # # Detect Blank (either too dark or too bright)
#     # if detect_blank(frame):
#     #     print("Camera tampering detected: Blank frame!")
#
#     # Detect Flash
#     if detect_flash(prev_frame, frame):
#         print("Camera tampering detected: Flash!")
#
#     prev_frame = frame  # Update previous frame
#
#     cv2.imshow("Camera Tampering Detection", frame)
#
#     if cv2.waitKey(30) & 0xFF == 27:  # Exit on 'ESC' key press
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import numpy as np
import cv2

cap = cv2.VideoCapture('final.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
kernel = np.ones((5, 5), np.uint8)

# Define the lookup table
lookuptable = np.empty((1, 256), np.uint8)
lookuptable[0, :] = [np.clip(pow(i / 255.0, 10) * 255.0, 0, 255) for i in range(256)]

flash_threshold = 115  # Adjust this threshold as needed
min_threshold = 500  # Minimum area threshold for considering a flash
max_threshold = 1000  # Maximum area threshold to exclude large bright areas

# Initialize the previous frame's brightness to None
prev_brightness = None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("End of frame")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness of the frame
    avg_brightness = np.mean(gray_frame)

    # Adjust the threshold dynamically based on the average brightness
    dynamic_threshold = max(200, avg_brightness + 50)

    # Apply the lookup table to enhance brightness
    res = cv2.LUT(gray_frame, lookuptable)

    # Apply thresholding to identify bright regions (potential flash)
    _, binary_img = cv2.threshold(res, dynamic_threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original frame for drawing bounding boxes
    img_with_boxes = frame.copy()

    flash_detected = False

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area == 0:
            continue

        if contour_area < min_threshold or contour_area > max_threshold:  # Filter based on area
            continue

        x, y, w, h = cv2.boundingRect(contour)
        white_pixel_percentage = (cv2.countNonZero(binary_img[y:y + h, x:x + w]) / contour_area) * 100
        print("white_pixel_percentage: ", white_pixel_percentage)
        if white_pixel_percentage > flash_threshold:
            flash_detected = True
            # Draw a green bounding box around the flash area
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # prev_brightness = current_brightness  # Update the previous brightness

    # Process the frame for tampering detection
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, kernel, iterations=2)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tampering_detected = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # print(w, h)
        if w >= 100 or h >= 100:
            contour_area = w * h
            if contour_area >= int(frame.shape[0] * frame.shape[1]) / 3:
                tampering_detected = True

    if tampering_detected:
        cv2.putText(frame, "TAMPERING DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if flash_detected:
        # Draw bounding box for flash (you can adjust this part based on your needs)
        cv2.putText(frame, "FLASH DETECTED", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        flash_mask = cv2.absdiff(fgmask, gray_frame)
        flash_contours, _ = cv2.findContours(flash_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in flash_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 20 or h >= 20:  # Set appropriate size threshold for flash
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()

