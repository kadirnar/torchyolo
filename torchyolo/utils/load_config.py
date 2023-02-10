from torchyolo.utils.config_utils import get_config


def load_parameters(config_path: str):
    config_path = config_path
    config = get_config(config_path)
    output_path = config.DATA_CONFIG.OUTPUT_PATH
    conf = config.DETECTOR_CONFIG.CONF_TH
    iou = config.DETECTOR_CONFIG.IOU_TH
    image_size = config.DETECTOR_CONFIG.IMAGE_SIZE
    device = config.DETECTOR_CONFIG.DEVICE
    save = config.DATA_CONFIG.SAVE
    show = config.DATA_CONFIG.SHOW
    return output_path, conf, iou, image_size, device, save, show
