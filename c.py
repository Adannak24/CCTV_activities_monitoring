import cv2
# from simple_facerec import SimpleFacerec
from s_f import FaceRecognitionClass
name = "UNKNOWN"
class FaceDetector:
    def __init__(self, encoding_images_path):
        self.sfr = FaceRecognitionClass()
        self.sfr.load_encodings(encoding_images_path)

    def register_face(self, img_path):
        self.sfr.append_face(img_path)

    def detect_faces(self, frame):
        global name
        face_locations, face_names = self.sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        return frame, name
