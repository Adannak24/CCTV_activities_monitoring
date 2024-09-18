import pickle
import face_recognition
import cv2
import numpy as np
import os
import glob


class FaceRecognitionClass:
    def __init__(self, encodings_file='Encodings/encodings.pkl', frame_resizing=1.0):
        self.encodings_file = encodings_file
        self.frame_resizing = frame_resizing
        self.known_face_encodings = []
        self.known_face_names = []
        # self.load_encodings("Encodings/encodings.pkl")

    def save_encodings(self):
        """
        Save face encodings and names to the encodings file.
        """
        try:
            with open(self.encodings_file, 'wb') as file:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, file)
            print("Encodings saved successfully.")
        except Exception as e:
            print(f"Error saving encodings: {e}")

    def delete_user_encoding(self, name_to_remove):
        """
        Delete a user's encoding from the encodings file.
        :param name_to_remove: The name of the user whose encoding should be removed.
        """
        print(self.known_face_names)
        if name_to_remove in self.known_face_names:
            index = self.known_face_names.index(name_to_remove)
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            self.save_encodings()
            print(f"Encoding for {name_to_remove} deleted successfully.")
        else:
            print(f"No encoding found for {name_to_remove}.")

    def load_encodings(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print("Encodings loaded from file.")
        else:
            self.load_encoding_images("images/")
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print("Encodings loaded from file.")

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        self.save_encodings()

    def append_face(self, image_path):
        """
        Append a new face encoding to the existing encodings file
        :param image_path: Path to the new face image
        """
        # Load existing encodings
        self.load_encodings('Encodings/encodings.pkl')

        # Read new image and get encoding
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        basename = os.path.basename(image_path)
        filename, ext = os.path.splitext(basename)
        img_encoding = face_recognition.face_encodings(rgb_img)[0]
        print("img_encoding", img_encoding)
        # Append new encoding and name
        self.known_face_encodings.append(img_encoding)
        self.known_face_names.append(filename)

        # Save updated encodings
        self.save_encodings()

    def detect_known_faces(self, frame):
        """
        Detect faces in the provided frame and match them against known faces.
        :param frame: The video frame to process.
        :return: Tuple of (face_locations, face_names)
        """
        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Adjust face locations for the resizing
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names




# Example usage
# face_recognition_system = FaceRecognitionClass()
# face_recognition_system.load_encodings("Encodings/encodings.pkl")
# frame = cv2.imread('Ratan_tata.jpeg')  # Load a sample frame
# locations, names = face_recognition_system.detect_known_faces(frame)
# print("Detected locations:", locations)
# print("Detected names:", names)
# face_recognition_system.delete_user_encoding("mukush ambani")
# face_recognition_system.append_face("Ratan_tata.jpeg")
