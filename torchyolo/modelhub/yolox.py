from yoloxdetect import YoloxDetector

from torchyolo.modelhub.basemodel import YoloDetectionModel


class YoloxDetectionModel(YoloDetectionModel):
    def load_model(self):
        model = YoloxDetector(self.model_path, config_path=self.config_path, device=self.device, hf_model=False)
        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        model.save = self.save
        model.show = self.show
        self.model = model

    def predict(self, image):
        self.model.predict(image_path=image, image_size=self.image_size)
