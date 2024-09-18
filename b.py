import cv2
import numpy as np

class CameraTamperingDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.kernel = np.ones((5, 5), np.uint8)
        self.lookuptable = np.empty((1, 256), np.uint8)
        self.lookuptable[0, :] = [np.clip(pow(i / 255.0, 10) * 255.0, 0, 255) for i in range(256)]

    def detect_tampering(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)
        dynamic_threshold = max(200, avg_brightness + 50)
        res = cv2.LUT(gray_frame, self.lookuptable)
        _, binary_img = cv2.threshold(res, dynamic_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flash_detected = False
        min_threshold = 500
        max_threshold = 5000
        flash_threshold = 115

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_threshold or contour_area > max_threshold:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            white_pixel_percentage = (cv2.countNonZero(binary_img[y:y + h, x:x + w]) / contour_area) * 100
            print(white_pixel_percentage)
            if white_pixel_percentage > flash_threshold:
                flash_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.erode(fgmask, self.kernel, iterations=2)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tampering_detected = False

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 100 or h >= 100:
                contour_area = w * h
                if contour_area >= int(frame.shape[0] * frame.shape[1]) / 3:
                    tampering_detected = True

        if tampering_detected:
            cv2.putText(frame, "TAMPERING DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # return frame, True

        if flash_detected:
            cv2.putText(frame, "FLASH DETECTED", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # return frame, True
        return frame, tampering_detected, flash_detected