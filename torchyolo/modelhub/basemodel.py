from typing import Optional

import numpy as np
import torch


class YoloDetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        image_size: int = 640,
    ):
        """
        Init object detection model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            iou_threshold: float
                All predictions with IoU < iou_threshold will be discarded
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            image_size: int
                Inference input size.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.show = False
        self.save = True
        if self.save:
            self.save_path = "output"
            self.output_file_name = "prediction_visual"

        # automatically load model if load_at_init is True
        self.load_model()

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        raise NotImplementedError()

    def predict(self, image: np.ndarray):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        raise NotImplementedError()
