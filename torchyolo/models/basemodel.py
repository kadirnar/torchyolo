from typing import Any, Optional

import numpy as np
import torch


class YoloDetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        image_size: int = None,
        load_at_init: bool = True,
    ):
        """
        Init object detection/instance segmentation model.
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
            load_at_init: bool
                If True, automatically loads the model at initalization
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = model
        self.device = device
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        # automatically load model if load_at_init is True
        if load_at_init:
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
