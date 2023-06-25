from dataclasses import dataclass
from pathlib import Path


# artifacts entity 
@dataclass
class DataIngestionArtifact:
    train_path:str
    test_path:str


@dataclass
class DataValidationArtifact:
    validation_status: bool


