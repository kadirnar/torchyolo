from torchyolo.automodel import AutoDetectionModel


class YoloHub:
    def __init__(self, config_path: str = None, model_type: str = "yolov5", model_path: str = None):
        self.config_path = config_path
        self.model_type = model_type
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model = AutoDetectionModel.from_pretrained(
            config_path=self.config_path,
            model_type=self.model_type,
            model_path=self.model_path,
        )

        return self.model

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

    def predict(self, source: str):
        prediction = self.model.predict(source=source)
        return prediction

    def tracker_predict(
        self,
        source: str = None,
        tracker_type: str = None,
        tracker_weight_path: str = None,
        tracker_config_path: str = None,
    ):
        tracker_prediction = self.model.tracker_predict(
            source=source,
            tracker_type=tracker_type,
            tracker_weight_path=tracker_weight_path,
            tracker_config_path=tracker_config_path,
        )
        return tracker_prediction


def tracker_predict(
    config_path: str = "torchyolo/configs/default_config.yaml",
    model_type: str = "yolov5",
    model_path: str = "yolov5s.pt",
    source: str = "test.mp4",
    tracker_type: str = "NORFAIR",
    tracker_weight_path: str = None,
    tracker_config_path: str = "torchyolo/configs/tracker/norfair_track.yaml",
):
    model = YoloHub(
        config_path=config_path,
        model_type=model_type,
        model_path=model_path,
    )

    if tracker_type == "STRONGSORT":
        result = model.tracker_predict(
            source=source,
            tracker_type=tracker_type,
            tracker_weight_path=tracker_weight_path,
            tracker_config_path=tracker_config_path,
    )    
    else:     
        result = model.tracker_predict(
            source=source,
            tracker_type=tracker_type,
            tracker_config_path=tracker_config_path,
        )
        
    return result


def predict(
    config_path: str = "torchyolo/configs/default_config.yaml",
    model_type: str = "yolov5",
    model_path: str = "yolov5s.pt",
    source: str = "test.mp4",
):
    model = YoloHub(
        config_path=config_path,
        model_type=model_type,
        model_path=model_path,
    )

    result = model.predict(source=source)
    return result
