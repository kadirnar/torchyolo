from pathlib import Path
from typing import Optional

from torchyolo.utils.config_utils import get_config

DEFAULT_BYTETRACK_CONFIG_PATH = "trackerhub/configs/byte_track.yaml"
DEFAULT_OCSORT_CONFIG_PATH = "trackerhub/configs/oc_sort.yaml"
DEFAULT_NORFAIR_CONFIG_PATH = "trackerhub/configs/norfair_track.yaml"
DEFAULT_STRONGSORT_CONFIG_PATH = "trackerhub/configs/strong_sort.yaml"
DEFAULT_SORT_CONFIG_PATH = "trackerhub/configs/sort_track.yaml"


def create_tracker(
    tracker_type,
    tracker_config_path,
    tracker_weight_path: Optional[str] = None,
    device: Optional[str] = "cpu",
    half: Optional[bool] = False,
    conf_th: Optional[str] = 0.05,
    iou_th: Optional[str] = 0.05,
) -> object:
    if tracker_type == "OC_SORT":
        try:
            from ocsort.ocsort import OCSort

            if tracker_config_path is None:
                config_path = DEFAULT_OCSORT_CONFIG_PATH
            else:
                config_path = tracker_config_path

            config = get_config(config_path)
            oc_sort = OCSort(
                det_thresh=conf_th,
                max_age=config.OC_SORT.MAX_AGE,
                min_hits=config.OC_SORT.MIN_HITS,
                iou_threshold=iou_th,
                delta_t=config.OC_SORT.DELTA_T,
                asso_func=config.OC_SORT.ASSO_FUNC,
                inertia=config.OC_SORT.INERTIA,
                use_byte=config.OC_SORT.USE_BYTE,
            )
            return oc_sort

        except ImportError:
            raise ImportError("Please install ocsort: pip install ocsort")

    elif tracker_type == "BYTE_TRACK":
        try:
            from bytetracker.byte_tracker import BYTETracker

            if tracker_config_path is None:
                config_path = DEFAULT_BYTETRACK_CONFIG_PATH
            else:
                config_path = tracker_config_path

            config = get_config(config_path)

            byte_tracker = BYTETracker(
                track_thresh=conf_th,
                track_buffer=config.BYTE_TRACK.TRACK_BUFFER,
                frame_rate=config.BYTE_TRACK.FRAME_RATE,
            )
            return byte_tracker

        except ImportError:
            raise ImportError("Please install bytetracker: pip install bytetracker")

    elif tracker_type == "NORFAIR_TRACK":
        try:
            from norfair_tracker.norfair import NorFairTracker

            if tracker_config_path is None:
                config_path = DEFAULT_NORFAIR_CONFIG_PATH
            else:
                config_path = tracker_config_path

            config = get_config(config_path)
            norfair_tracker = NorFairTracker(
                distance_function=config.NORFAIR_TRACK.DISTANCE_FUNCTION,
                distance_threshold=config.NORFAIR_TRACK.DISTANCE_THRESHOLD,
                hit_counter_max=config.NORFAIR_TRACK.HIT_COUNTER_MAX,
                initialization_delay=config.NORFAIR_TRACK.INITIALIZATION_DELAY,
                pointwise_hit_counter_max=config.NORFAIR_TRACK.POINTWISE_HIT_COUNTER_MAX,
                detection_threshold=config.NORFAIR_TRACK.DETECTION_THRESHOLD,
                past_detections_length=config.NORFAIR_TRACK.PAST_DETECTIONS_LENGTH,
                reid_distance_threshold=config.NORFAIR_TRACK.REID_DISTANCE_THRESHOLD,
                reid_hit_counter_max=config.NORFAIR_TRACK.REID_HIT_COUNTER_MAX,
            )
            return norfair_tracker
        except ImportError:
            raise ImportError("Please install norfair: pip install norfair-tracker")

    elif tracker_type == "SORT_TRACK":
        try:
            from sort.tracker import SortTracker

            if tracker_config_path is None:
                config_path = DEFAULT_SORT_CONFIG_PATH
            else:
                config_path = tracker_config_path

            config = get_config(config_path)
            sort_tracker = SortTracker(
                max_age=config.SORT_TRACK.MAX_AGE,
                min_hits=config.SORT_TRACK.MIN_HITS,
                iou_threshold=iou_th,
            )
            return sort_tracker

        except ImportError:
            raise ImportError("Please install sort: pip install sort-track")

    elif tracker_type == "STRONG_SORT":
        try:
            from strongsort.strong_sort import StrongSORT

            if tracker_config_path is None:
                config_path = DEFAULT_STRONGSORT_CONFIG_PATH
            else:
                config_path = tracker_config_path

            config = get_config(config_path)
            strong_sort = StrongSORT(
                tracker_weight_path,
                device,
                half,
                max_dist=config.STRONG_SORT.MAX_DIST,
                max_iou_distance=config.STRONG_SORT.MAX_IOU_DISTANCE,
                max_age=config.STRONG_SORT.MAX_AGE,
                n_init=config.STRONG_SORT.N_INIT,
                nn_budget=config.STRONG_SORT.NN_BUDGET,
                mc_lambda=config.STRONG_SORT.MC_LAMBDA,
                ema_alpha=config.STRONG_SORT.EMA_ALPHA,
            )
            return strong_sort

        except ImportError:
            raise ImportError("Please install strongsort: pip install strongsort")

    else:
        raise ValueError(f"No such tracker: {tracker_type}")


def load_tracker(config_path: str) -> object:
    """
    This function is used to track objects in a video using yolov5 and strong sort.
    Args:
        video_path: video path(str)
    """
    config = get_config(config_path)
    tracker_module = create_tracker(
        tracker_type=config.TRACKER_CONFIG.TRACKER_TYPE,
        tracker_weight_path=Path(config.TRACKER_CONFIG.WEIGHT_PATH),
        tracker_config_path=config.TRACKER_CONFIG.CONFIG_PATH,
        device=config.DETECTOR_CONFIG.DEVICE,
        half=config.DETECTOR_CONFIG.HALF,
        conf_th=config.DETECTOR_CONFIG.CONF_TH,
        iou_th=config.DETECTOR_CONFIG.IOU_TH,
    )
    return tracker_module
