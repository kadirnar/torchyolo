from torchyolo.automodel import AutoDetectionModel


class YoloPredictor:
    def __init__(self, model_type="yolov5", model_path="yolov5s.pt", device="cpu", image_size=640):
        self.model_type = model_type
        self.model_path = model_path
        self.config_path = "torchyolo.configs.yolox.yolox_m"  # yolox_nano.py
        self.device = device
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.image_size = image_size
        self.save = True
        self.show = False

        # Load Model
        self.load_model()

    def load_model(self):
        model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model_path=self.model_path,
            config_path=self.config_path,
            device=self.device,
            confidence_threshold=self.conf_thres,
            iou_threshold=self.iou_thres,
            image_size=self.image_size,
        )
        model.save = self.save
        model.show = self.show
        self.model = model

    def predict(self, image):
        return self.model.predict(image)


if __name__ == "__main__":
    predictor = YoloPredictor(model_type="yolov5", model_path="yolov5s.pt", device="cpu", image_size=640)
    image = "data/highway.jpg"
    result = predictor.predict(image)
