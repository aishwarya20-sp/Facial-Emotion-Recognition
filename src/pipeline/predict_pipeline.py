import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class PredictPipeline:
    def __init__(self):
        model_path = "artifacts/model.keras"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = load_model(model_path)

        self.emotion_labels = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise"
        }

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def predict_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float32") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            prediction = self.model.predict(roi_gray, verbose=0)

            emotion_index = np.argmax(prediction)
            confidence = np.max(prediction)

            emotion = f"{self.emotion_labels[emotion_index]} ({confidence:.2f})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return frame


def run_webcam():
    predictor = PredictPipeline()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to access webcam.")
        return

    print("\nPress 'q' to quit.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output_frame = predictor.predict_emotion(frame)

        cv2.imshow("Facial Emotion Recognition", output_frame)

        if cv2.waitKey(1) == ord("q"):
            break
    os.makedirs("results", exist_ok=True)

    cv2.imwrite(
    "results/final_prediction.jpg",
    output_frame
)

    print("Prediction image saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()