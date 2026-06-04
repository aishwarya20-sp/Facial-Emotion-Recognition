import os
from dataclasses import dataclass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.keras")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(7, activation="softmax"))

        return model

    def initiate_model_trainer(self, train_generator, test_generator):
        os.makedirs("artifacts", exist_ok=True)

        model = self.get_model()

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        checkpoint = ModelCheckpoint(
            self.config.trained_model_file_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )

        history = model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=50,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )

        return model, history