import cv2
import yolov5
from modelhub.basemodel import YoloDetectionModel


class Yolov5DetectionModel(YoloDetectionModel):
    def load_model(self):
        model = yolov5.load(self.model_path, device=self.device)
        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        self.model = model

    def predict(self, image):
        prediction = self.model(image, size=self.image_size)

        for _, image_predictions_in_xyxy_format in enumerate(prediction.xyxy):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_name = self.model.names[int(pred[5])]
                category_id = pred[5]
                labels = f"{category_name} {score:.2f}"
