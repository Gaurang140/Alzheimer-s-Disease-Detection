
from dataclasses import dataclass
from alzheimer_disease.entity import artifacts_entity 
from alzheimer_disease.entity import config_entity
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging 
from alzheimer_disease.config import ModelManager
from alzheimer_disease.utils.main_utils import load_object,save_object
import sys 
import shutil

class ModelPusher:

    def __init__(self,model_pusher_config:config_entity.ModelPusherConfig,
    model_trainer_artifact:artifacts_entity.ModelTrainerArtifcats):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelManager(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise AlzException(e, sys)

    def initiate_model_pusher(self,)->artifacts_entity.ModelPusherArtifact:
        try:
            #load object
            logging.info(f"Loading transformer model and target encoder")
        
            model = load_object(file_path=self.model_trainer_artifact.model_dir)
           

            #model pusher dir
            logging.info(f"Saving model into model pusher directory")
       
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
           
            
    


            #saved model dir
            logging.info(f"Saving model in saved model dir")
         
            model_path=self.model_resolver.get_latest_save_model_path()
         

            save_object(file_path=model_path, obj=model)
            

            model_pusher_artifact =artifacts_entity.ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
             saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise AlzException(e, sys)


        




