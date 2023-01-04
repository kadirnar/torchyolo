from typing import Optional

MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "yolov5": "Yolov5DetectionModel",
}


class AutoDetectionModel:
    def from_pretrained(
        model_type: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        image_size: int = None,
        **kwargs,
    ):
        """
        Loads a DetectionModel from given path.
        Args:
            model_type: str
                Name of the detection framework (example: "yolov5", "mmdet", "detectron2")
            model_path: str
                Path of the detection model (ex. 'model.pt')
            config_path: str
                Path of the config file (ex. 'mmdet/configs/cascade_rcnn_r50_fpn_1x.py')
            device: str
                Device, "cpu" or "cuda:0"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        Returns:
            Returns an instance of a DetectionModel
        Raises:
            ImportError: If given {model_type} framework is not installed
        """

        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        DetectionModel = getattr(
            __import__(f"torchyolo.modelhub.{model_type}", fromlist=[model_class_name]), model_class_name
        )
        return DetectionModel(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            **kwargs,
        )
