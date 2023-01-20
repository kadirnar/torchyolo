MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "yolov5": "Yolov5DetectionModel",
    "yolov7": "Yolov7DetectionModel",
    "yolov8": "Yolov8DetectionModel",
}


class AutoDetectionModel:
    def from_pretrained(
        config_path: str,
        model_type: str = "yolov5",
    ):

        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        DetectionModel = getattr(
            __import__(f"torchyolo.modelhub.{model_type}", fromlist=[model_class_name]), model_class_name
        )

        return DetectionModel(
            config_path=config_path,
        )
