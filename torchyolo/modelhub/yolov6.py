from yolov6 import YOLOV6

from torchyolo.modelhub.basemodel import YoloDetectionModel


class Yolov6DetectionModel(YoloDetectionModel):
    def load_model(self):
        model = YOLOV6(self.model_path, device=self.device)
        model.font_path = "torchyolo/configs/yolov6/Arial.ttf"
        model.conf_thres = self.confidence_threshold
        model.iou_thresh = self.iou_threshold
        model.save_img = self.save
        model.show_img = self.show
        self.model = model

    def predict(self, image):
        self.model.predict(source=image, img_size=self.image_size, yaml=self.yaml_file)
