from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        print("\nStarting Training Pipeline...\n")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_datagen, test_datagen = transformation.get_data_transformer_object()

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=64,
            class_mode="categorical",
            shuffle=True
        )

        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=64,
            class_mode="categorical",
            shuffle=False
        )

        print("\nData Transformation Completed\n")

        trainer = ModelTrainer()
        model, history = trainer.initiate_model_trainer(
            train_generator,
            test_generator
        )

        print("\nModel Training Completed")
        print("Model saved at: artifacts/model.keras\n")


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()