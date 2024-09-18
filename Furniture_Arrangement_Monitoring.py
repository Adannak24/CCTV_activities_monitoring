import cv2
from skimage.metrics import structural_similarity as compare_ssim

# Load initial classroom setup image
initial_image = cv2.imread("initial_setup.jpg", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(initial_image, gray_frame, full=True)
    diff = (diff * 255).astype("uint8")

    if score < 0.8:  # Threshold for anomaly detection
        print("Anomaly detected in furniture arrangement!")

    cv2.imshow("Live", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
