import cv2
from datetime import datetime, time


# Function to check if the current time is within exam hours
def is_exam_time():
    now = datetime.now().time()
    exam_start = time(11, 0)  # 9:00 AM
    exam_end = time(12, 0)  # 12:00 PM
    return exam_start <= now <= exam_end


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    if not is_exam_time():
        continue

    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    # Check if movement is detected (non-zero values in the mask)
    if cv2.countNonZero(fgmask) > 5000:  # Threshold can be adjusted
        print("Unauthorized movement detected during exam time!")

    # Display the result
    cv2.imshow('Unauthorized Movement Detection', fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
