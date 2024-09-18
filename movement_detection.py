import cv2


class MotionDetection:
    def __init__(self, threshold=100000):
        # Initialize the background subtractor and set a threshold for movement detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.threshold = threshold

    def detect(self, frame):
        # Apply the background subtractor to get the foreground mask
        fgmask = self.fgbg.apply(frame)
        print(cv2.countNonZero(fgmask))
        # Check if movement is detected (based on non-zero values in the mask)
        if cv2.countNonZero(fgmask) > self.threshold:
            print("Movement detected!")
            return frame, True

        return frame, False  # Return the mask so it can be displayed or further processed