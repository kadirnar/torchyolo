from typing import Optional

from torchyolo.utils.config_utils import get_config

MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "yolov5": "Yolov5DetectionModel",
    "yolov6": "Yolov6DetectionModel",
    "yolov7": "Yolov7DetectionModel",
    "yolov8": "Yolov8DetectionModel",
    "yolox": "YoloxDetectionModel",
}


class AutoDetectionModel:
    def from_pretrained(
        config_path: str,
    ):

        config = get_config(config_path)
        model_type = config.DETECTOR_CONFIG.DETECTOR_TYPE
        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        DetectionModel = getattr(
            __import__(f"torchyolo.modelhub.{model_type}", fromlist=[model_class_name]), model_class_name
        )

        return DetectionModel(
            config_path=config_path,
        )
