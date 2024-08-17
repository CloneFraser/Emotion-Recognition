import haar_cascade
import cv2


class Frame:
    def __init__(self, CNN, CNN_model):
        self.haar_classifier = haar_cascade.HaarCascade()
        self.CNN = CNN
        self.CNN_model = CNN_model

    def get_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_classifier.detect_faces(gray_image=gray)

        prediction = "Null"

        for (x, y, w, h) in faces:
            prepped_img = self.CNN.model_preprocessing(gray, x, y, w, h)
            prediction = self.CNN.model_prediction(self.CNN_model, prepped_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_8, 0)
            cv2.putText(frame, prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return frame, prediction

