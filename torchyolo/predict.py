from automodel import AutoDetectionModel

from torchyolo.utils.config_utils import get_config


def model_prediction(config_file):
    config = get_config(config_file)
    model = AutoDetectionModel.from_pretrained(
        model_type=config.DETECTOR_CONFIG.MODEL_TYPE,
        model_path=config.DETECTOR_CONFIG.MODEL_PATH,
        config_path=config.DETECTOR_CONFIG.CONFIG_PATH,
        device=config.DETECTOR_CONFIG.DEVICE,
        confidence_threshold=config.DETECTOR_CONFIG.CONFIDENCE_THRESHOLD,
        iou_threshold=config.DETECTOR_CONFIG.IOU_THRESHOLD,
        image_size=config.DETECTOR_CONFIG.IMAGE_SIZE,
    )
    model.show = config.DETECTOR_CONFIG.SHOW
    model.save = config.DETECTOR_CONFIG.SAVE

    model.predict(image=config.FILE_CONFIG.IMAGE_PATH)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--config_file", type=str, default="torchyolo/configs/default_config.yaml")
    args = args.parse_args()
    model_prediction(args.config_file)
