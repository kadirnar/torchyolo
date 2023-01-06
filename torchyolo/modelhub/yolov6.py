import cv2
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.utils.cv import visualize_object_predictions
from yolov6 import YOLOV6

from torchyolo.modelhub.basemodel import YoloDetectionModel


class Yolov6DetectionModel(YoloDetectionModel):
    def load_model(self):
        model = YOLOV6(self.model_path, device=self.device)
        model.torchyolo = True
        model.font_path = "torchyolo/configs/yolov6/Arial.ttf"
        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        model.save_img = self.save
        model.show_img = self.show
        self.model = model

    def predict(self, image, yaml_file="torchyolo/configs/yolov6/coco.yaml"):
        object_prediction_list = []
        predictions, class_names = self.model.predict(source=image, img_size=self.image_size, yaml=yaml_file)
        for *xyxy, conf, cls in reversed(predictions.cpu().detach().numpy()):
            x1, y1, x2, y2 = (
                int(xyxy[0]),
                int(xyxy[1]),
                int(xyxy[2]),
                int(xyxy[3]),
            )
            bbox = [x1, y1, x2, y2]
            score = conf
            category_id = int(cls)
            category_name = class_names[category_id]

            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=int(category_id),
                score=score,
                category_name=category_name,
            )
            object_prediction_list.append(object_prediction)

        prediction_result = PredictionResult(
            object_prediction_list=object_prediction_list,
            image=image,
        )
        if self.save:
            prediction_result.export_visuals(export_dir=self.save_path, file_name=self.output_file_name)

        if self.show:
            image = cv2.imread(image)
            output_image = visualize_object_predictions(image=image, object_prediction_list=object_prediction_list)
            cv2.imshow("Prediction", output_image["image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return prediction_result
