import cv2
from tqdm import tqdm

from torchyolo.tracker.tracker_zoo import load_tracker
from torchyolo.utils.config_utils import get_config
from torchyolo.utils.dataset import LoadData, create_video_writer
from torchyolo.utils.object_vis import video_vis


class Yolov5DetectionModel:
    def __init__(
        self,
        config_path: str,
    ):
        self.load_config(config_path)
        self.load_model()

    def load_config(self, config_path: str):
        self.config_path = config_path
        config = get_config(config_path)
        self.output_path = config.DATA_CONFIG.OUTPUT_PATH
        self.conf = config.DETECTOR_CONFIG.CONF_TH
        self.iou = config.DETECTOR_CONFIG.IOU_TH
        self.image_size = config.DETECTOR_CONFIG.IMAGE_SIZE
        self.device = config.DETECTOR_CONFIG.DEVICE
        self.save = config.DATA_CONFIG.SAVE
        self.show = config.DATA_CONFIG.SHOW

    def load_model(self, model_path: str = "yolov5n.pt"):
        try:
            import yolov5

            model = yolov5.load(model_path, device=self.device)
            model.conf = self.conf
            model.iou = self.iou
            self.model = model

        except ImportError:
            raise ImportError('Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

    def predict(
        self,
        source: str = None,
        tracker_type: str = None,
        tracker_weight_path: str = None,
        tracker_config_path: str = None,
    ):
        if tracker_type == "STRONGSORT":
            tracker_module = load_tracker(
                config_path=self.config_path,
                tracker_type=tracker_type,
                tracker_weight_path=tracker_weight_path,
                tracker_config_path=tracker_config_path,
            )

        else:
            tracker_module = load_tracker(
                config_path=self.config_path,
                tracker_type=tracker_type,
                tracker_config_path=tracker_config_path,
            )

        tracker_outputs = [None]
        dataset = LoadData(source)
        video_writer = create_video_writer(video_path=source, output_path=self.output_path)
        for img_src, img_path, vid_cap in tqdm(dataset):
            results = self.model(img_src, size=self.image_size)
            for image_id, prediction in enumerate(results.pred):
                if tracker_type is not None:
                    tracker_outputs[image_id] = tracker_module.update(prediction.cpu(), img_src)
                    for output in tracker_outputs[image_id]:
                        bbox, track_id, category_id, score = (
                            output[:4],
                            int(output[4]),
                            output[5],
                            output[6],
                        )
                        category_name = self.model.names[int(category_id)]
                        label = f"Id:{track_id} {category_name} {float(score):.2f}"

                        if self.save or self.show:
                            frame = video_vis(
                                bbox=bbox,
                                label=label,
                                frame=img_src,
                                object_id=int(category_id),
                            )
                            if self.save:
                                video_writer.write(frame)

                            if self.show:
                                cv2.imshow("frame", frame)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    break

                else:
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
                        label = f"{category_name} {score:.2f}"

                    frame = video_vis(
                        bbox=bbox,
                        label=label,
                        frame=img_src,
                        object_id=category_id,
                    )
                    if self.save:
                        video_writer.write(frame)

                    if self.show:
                        cv2.imshow("frame", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
