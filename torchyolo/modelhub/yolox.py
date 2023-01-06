import cv2
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.utils.cv import visualize_object_predictions
from yoloxdetect import YoloxDetector

from torchyolo.modelhub.basemodel import YoloDetectionModel


class YoloxDetectionModel(YoloDetectionModel):
    def load_model(self):
        model = YoloxDetector(self.model_path, config_path=self.config_path, device=self.device, hf_model=False)
        model.torchyolo = True
        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        model.save = self.save
        model.show = self.show
        self.model = model

    def predict(self, image, yaml_file=None):
        object_prediction_list = []
        predict_list = self.model.predict(image_path=image, image_size=self.image_size)
        boxes, scores, cls_ids, class_names = predict_list[0], predict_list[1], predict_list[2], predict_list[3]
        for i in range(len(boxes)):
            box = boxes[i]
            category_id = int(cls_ids[i])
            score = scores[i]
            if score < self.confidence_threshold:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            bbox = [x0, y0, x1, y1]
            category_name = class_names[category_id]

            object_prediction = ObjectPrediction(
                bbox=bbox,
                score=score,
                category_id=category_id,
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
