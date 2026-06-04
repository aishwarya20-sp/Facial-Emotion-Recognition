import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("notebook", "data", "train")
    test_path: str = os.path.join("notebook", "data", "test")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        train_path = self.config.train_path
        test_path = self.config.test_path

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train path not found: {train_path}")

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test path not found: {test_path}")

        print("Data Ingestion Completed")
        print("Train Path:", train_path)
        print("Test Path:", test_path)

        return train_path, test_path