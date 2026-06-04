from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@dataclass
class DataTransformationConfig:
    image_size: tuple = (48, 48)
    batch_size: int = 64
    color_mode: str = "grayscale"
    class_mode: str = "categorical"


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(
            rescale=1.0 / 255
        )

        return train_datagen, test_datagen