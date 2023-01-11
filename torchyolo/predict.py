from typing import Optional

from torchyolo.automodel import AutoDetectionModel


class YoloHub:
    def __init__(
        self,
        model_type: str = "yolov5",
        model_path: str = "yolov5s.pt",
        device: str = "cpu",
        image_size: int = 640,
        config_path: Optional[str] = "configs.yolox.yolox_s",
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.conf = 0.45
        self.iou = 0.45
        self.image_size = image_size
        self.save = True
        self.show = False
        self.model = None

        # Load Model
        self.load_model()

    def load_model(self):
        model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model_path=self.model_path,
            config_path=self.config_path,
            device=self.device,
            confidence_threshold=self.conf,
            iou_threshold=self.iou,
            image_size=self.image_size,
        )
        model.save = self.save
        model.show = self.show
        self.model = model
        return model

    def prediction(self, image, yaml_file=None):
        return self.model.predict(image, yaml_file=yaml_file)


if __name__ == "__main__":
    model = YoloHub(model_type="yolov8", model_path="yolov8n.pt", device="cuda:0", image_size=640)
    image = "data/highway.jpg"
    result = model.prediction(image)
