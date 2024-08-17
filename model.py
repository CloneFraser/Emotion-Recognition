import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Model:
    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        model.load_weights('model.h5')
        return model

    def model_preprocessing(self, grey_image, x, y, width, height):
        roi_grey = grey_image[y:y + height, x:x + width]
        resized_img = np.expand_dims(np.expand_dims(cv2.resize(roi_grey, (48, 48)), -1), 0)
        return resized_img

    def model_prediction(self, model, resized_img):
        emotion_dict = {0: "Angry", 1: "Sad", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        prediction = model.predict(resized_img)
        max_index = int(np.argmax(prediction))
        return emotion_dict[max_index]



