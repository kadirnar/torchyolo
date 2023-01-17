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

    def view_model_architecture(self, file_format: str = "pdf"):
        try:
            from torchview import draw_graph
        except:
            raise ImportError("Please install torchview: pip install torchview")

        if self.model_type == "yolov5" or self.model_type == "yolov7":
            model_arch = self.model.model

        elif self.model_type == "yolov8" or self.model_type == "yolox":
            model_arch = self.model.model.model

        elif self.model_type == "yolov6":
            model_arch = self.model.model.model.model

        model_graph = draw_graph(
            model_arch,
            input_size=(1, 3, 384, 640),
            expand_nested=True,
            depth=3,
        )

        model_graph.visual_graph.render(format=file_format)
        return model_graph



if __name__ == "__main__":
    model = YoloHub(model_type="yolov5", model_path="yolov5n.pt", device="cuda:0", image_size=640)
    result = model.predict("test.mp4")
