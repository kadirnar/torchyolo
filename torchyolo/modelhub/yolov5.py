import cv2
import yolov5
from torchyolo.utils.dataset import LoadData, create_video_writer
from torchyolo.modelhub.basemodel import YoloDetectionModel
from torchyolo.utils.object_vis import video_vis


from tqdm import tqdm

class Yolov5DetectionModel(YoloDetectionModel):
    def load_model(self):
        model = yolov5.load(self.model_path, device=self.device)
        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        self.model = model

    def predict(self, image, yaml_file=None, save=False, show=False):
        dataset = LoadData(image)
        video_writer = create_video_writer(
            video_path=image,
            output_path='output')
        for img_src, img_path, vid_cap in tqdm(dataset):
            results = self.model(img_src, augment=False)
            for index, prediction in enumerate(results.pred):
                for pred in prediction.cpu().detach().numpy():
                    x1, y1, x2, y2 = (
                        int(pred[0]),
                        int(pred[1]),
                        int(pred[2]),
                        int(pred[3]),
                    )
                    bbox = [x1, y1, x2, y2]
                    score = pred[4]
                    category_name = self.model.names[int(pred[5])]
                    category_id = int(pred[5])
                    label = f'{category_name} {score:.2f}'
                     
                frame = video_vis(
                    bbox=bbox,
                    label=label,
                    frame=img_src,
                    object_id=category_id,
                )
                if save:
                    video_writer.write(frame)
                    
                if show:
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
        video_writer.release()
