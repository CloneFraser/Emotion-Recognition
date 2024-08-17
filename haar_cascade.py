import cv2


class HaarCascade:
    def generate_haar_classifier(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade

    def detect_faces(self, gray_image):
        face_cascade = self.generate_haar_classifier()
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        return faces


