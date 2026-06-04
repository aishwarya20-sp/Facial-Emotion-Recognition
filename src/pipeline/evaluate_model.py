import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import (
    load_model
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


class ModelEvaluation:

    def __init__(self):

        self.model = load_model(
            "artifacts/model.keras"
        )

    def evaluate_model(self):

        test_datagen = (
            ImageDataGenerator(
                rescale=1./255
            )
        )

        test_generator = (
            test_datagen.flow_from_directory(
                "notebook/data/test",
                target_size=(48,48),
                color_mode="grayscale",
                batch_size=64,
                class_mode="categorical",
                shuffle=False
            )
        )

        loss, accuracy = (
            self.model.evaluate(
                test_generator
            )
        )

        print(
            f"\nTest Loss: {loss:.4f}"
        )

        print(
            f"Test Accuracy: {accuracy:.4f}"
        )

        y_pred = self.model.predict(
            test_generator
        )

        y_pred_classes = np.argmax(
            y_pred,
            axis=1
        )

        y_true = (
            test_generator.classes
        )

        class_names = list(
            test_generator.class_indices.keys()
        )

        print(
            classification_report(
                y_true,
                y_pred_classes,
                target_names=class_names
            )
        )

        cm = confusion_matrix(
            y_true,
            y_pred_classes
        )

        plt.figure(
            figsize=(8,6)
        )

        plt.imshow(cm)

        plt.colorbar()

        plt.title(
            "Confusion Matrix"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.show()


if __name__ == "__main__":

    evaluator = ModelEvaluation()

    evaluator.evaluate_model()