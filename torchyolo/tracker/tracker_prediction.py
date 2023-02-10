from torchyolo.tracker.tracker_zoo import load_tracker
from torchyolo.utils.config_utils import get_config


class TrackerPrediction:
    def __init__(
        self,
        config_path: str,
        tracker_type: str = None,
        tracker_weight_path: str = None,
        tracker_config_path: str = None,
    ):
        self.config_path = config_path
        self.tracker_type = tracker_type
        self.tracker_weight_path = tracker_weight_path
        self.tracker_config_path = tracker_config_path
        self.load_config()

    def load_config(self):
        config = get_config(self.config_path)
        self.output_path = config.DATA_CONFIG.OUTPUT_PATH
        self.conf = config.DETECTOR_CONFIG.CONF_TH
        self.iou = config.DETECTOR_CONFIG.IOU_TH
        self.image_size = config.DETECTOR_CONFIG.IMAGE_SIZE
        self.device = config.DETECTOR_CONFIG.DEVICE
        self.save = config.DATA_CONFIG.SAVE
        self.show = config.DATA_CONFIG.SHOW

    def predict(self):
        if self.tracker_type == "STRONGSORT":
            tracker_module = load_tracker(
                config_path=self.config_path,
                tracker_type=self.tracker_type,
                tracker_weight_path=self.tracker_weight_path,
                tracker_config_path=self.tracker_config_path,
            )

        else:
            tracker_module = load_tracker(
                config_path=self.config_path,
                tracker_type=self.tracker_type,
                tracker_weight_path=self.tracker_weight_path,
                tracker_config_path=self.tracker_config_path,
            )

        return tracker_module
