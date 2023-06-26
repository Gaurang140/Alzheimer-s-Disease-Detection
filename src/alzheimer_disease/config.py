import os
from typing import Optional


class ModelManager:
    def __init__(self, model_registry: str = "saved_models"):
        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)

    def get_latest_dir_path(self) -> Optional[str]:
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registry, f"{latest_dir_name}")
        except Exception as e:
            raise e

    def get_latest_save_dir_path(self) -> str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry, f"{0}")
            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registry, f"{latest_dir_num + 1}")
        except Exception as e:
            raise e

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir, "model")
        except Exception as e:
            raise e

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, "model")
        except Exception as e:
            raise e
