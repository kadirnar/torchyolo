from fire import Fire

from torchyolo import __version__ as torchyolo_version
from torchyolo.predict import predict, tracker_predict

torchyolo_app = {"version": torchyolo_version, "predict": predict, "tracker": tracker_predict}


def app():
    Fire(torchyolo_app)


if __name__ == "__main__":
    app()
