import cv2
from tqdm import tqdm
from yolov6.core.inferer import Inferer
from yolov6.helpers import check_img_size
from yolov6.utils.nms import non_max_suppression

from torchyolo.tracker.tracker_zoo import load_tracker
from torchyolo.utils.coco_classes import COCO_CLASSES
from torchyolo.utils.config_utils import get_config
from torchyolo.utils.dataset import LoadData, create_video_writer
from torchyolo.utils.object_vis import video_vis


class Yolov6DetectionModel:
    def __init__(
        self,
        config_path: str,
        model_path: str = "yolov6s.pt",
    ):

        self.model_path = model_path
        self.load_config(config_path)
        self.load_model()

    def load_config(self, config_path: str):
        self.config_path = config_path
        config = get_config(config_path)
        self.output_path = config.DATA_CONFIG.OUTPUT_PATH
        self.device = config.DETECTOR_CONFIG.DEVICE
        self.conf = config.DETECTOR_CONFIG.CONF_TH
        self.iou = config.DETECTOR_CONFIG.IOU_TH
        self.image_size = config.DETECTOR_CONFIG.IMAGE_SIZE
        self.save = config.DATA_CONFIG.SAVE
        self.show = config.DATA_CONFIG.SHOW
        self.hf_model = config.DETECTOR_CONFIG.HUGGING_FACE_MODEL
        self.yolov6_yaml_file = config.DETECTOR_CONFIG.YOLOV6_YAML_FILE

    def load_model(self):
        try:
            from yolov6 import YOLOV6

            model = YOLOV6(self.model_path, device=self.device, hf_model=self.hf_model)
            model.conf = self.conf
            model.iou = self.iou
            self.model = model.model

        except ImportError:
            raise ImportError('Please run "pip install yolov6detect" ' "to install YOLOv6 first for YOLOv6 inference.")

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
        imga_size = check_img_size(self.image_size, s=32)

        for img_src, img_path, vid_cap in tqdm(dataset):
            img, img_src = Inferer.precess_image(img_src, imga_size, 32, False)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim

            pred_results = self.model(img)
            det = non_max_suppression(pred_results, self.conf, self.iou, classes=None, agnostic=False, max_det=1000)[0]

            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()

            if tracker_type:
                for image_index, result in enumerate([det]):
                    tracker_outputs[image_index] = tracker_module.update(result.cpu(), img_src)
                    for output in tracker_outputs[image_index]:
                        bbox, track_id, category_id, score = (
                            output[:4],
                            int(output[4]),
                            output[5],
                            output[6],
                        )
                        category_name = COCO_CLASSES[int(category_id)]
                        label = f"Id:{track_id} {category_name} {float(score):.2f}"

                        if self.save or self.show:
                            img_src = video_vis(
                                bbox=bbox,
                                label=label,
                                frame=img_src,
                                object_id=int(category_id),
                            )
                if self.save:
                    video_writer.write(img_src)
            else:
                for *xyxy, conf, cls in det:
                    label = f"{COCO_CLASSES[int(cls)]} {float(conf):.2f}"
                    frame = video_vis(
                        bbox=xyxy,
                        label=label,
                        frame=img_src,
                        object_id=int(cls),
                    )
                    if self.save:
                        if source.endswith(".mp4"):
                            video_writer.write(frame)
                        else:
                            cv2.imwrite("output.jpg", frame)
