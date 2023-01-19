from torchyolo.automodel import AutoDetectionModel
from torchyolo.utils.config_utils import get_config


class YoloHub:
    def __init__(self, config_path: str):
        self.load_config(config_path)

    def load_config(self, config_path: str):
        self.config_path = config_path
        config = get_config(config_path)
        self.input_path = config.DATA_CONFIG.INPUT_PATH
        self.output_path = config.DATA_CONFIG.OUTPUT_PATH
        self.model_type = config.DETECTOR_CONFIG.DETECTOR_TYPE
        self.model_path = config.DETECTOR_CONFIG.MODEL_PATH
        self.device = config.DETECTOR_CONFIG.DEVICE
        self.conf = config.DETECTOR_CONFIG.CONF_TH
        self.iou = config.DETECTOR_CONFIG.IOU_TH
        self.image_size = config.DETECTOR_CONFIG.IMAGE_SIZE
        self.save = config.DATA_CONFIG.SAVE
        self.show = config.DATA_CONFIG.SHOW

        # Load Model
        self.load_model()

    def load_model(self):
        model = AutoDetectionModel.from_pretrained(
            config_path=self.config_path,
        )
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

    def predict(self, tracker: bool = False):
        return self.model.predict(tracker)


if __name__ == "__main__":
    model = YoloHub(config_path="torchyolo/default_config.yaml")
    result = model.predict(tracker=True)
