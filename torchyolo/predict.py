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
    
    def models_view(self):
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
            
        model_graph = draw_graph(model_arch, 
                                    input_size=(1, 3, 352, 352), 
                                    expand_nested=True, 
                                    depth=3,
                                )
        model_graph.visual_graph.view()

if __name__ == "__main__":
    model = YoloHub(model_type="yolov7", model_path="yolov7.pt", device="cuda:0", image_size=640)
    image = "data/highway.jpg"
    result = model.models_view()
