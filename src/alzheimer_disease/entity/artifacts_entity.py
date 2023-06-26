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



@dataclass
class ModelTrainerArtifcats:
    model_dir :str
    test_path:str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float




@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str 
    saved_model_dir:str



