import os

import yaml
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert os.path.isfile(config_file)
            with open(config_file, "r") as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, "r") as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file: str = None) -> YamlParser:
    """
    This function is used to load config.
    Args:
        config_file: config file path
    Returns:
        config: config
    """
    config = YamlParser(config_file=config_file)
    config.merge_from_file(config_file)
    return config


def load_config(config_path: str):
    config_path = config_path
    config = get_config(config_path)
    input_path = config.DATA_CONFIG.INPUT_PATH
    output_path = config.DATA_CONFIG.OUTPUT_PATH
    model_type = config.DETECTOR_CONFIG.DETECTOR_TYPE
    model_path = config.DETECTOR_CONFIG.MODEL_PATH
    device = config.DETECTOR_CONFIG.DEVICE
    conf = config.DETECTOR_CONFIG.CONF_TH
    iou = config.DETECTOR_CONFIG.IOU_TH
    image_size = config.DETECTOR_CONFIG.IMAGE_SIZE
    save = config.DATA_CONFIG.SAVE
    show = config.DATA_CONFIG.SHOW

    return {
        "config_path": config_path,
        "input_path": input_path,
        "output_path": output_path,
        "model_type": model_type,
        "model_path": model_path,
        "device": device,
        "conf": conf,
        "iou": iou,
        "image_size": image_size,
        "save": save,
        "show": show,
    }
