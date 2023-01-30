from fire import Fire

from torchyolo import __version__ as torchyolo_version
from torchyolo.predict import main

torchyolo_app = {
    "version": torchyolo_version,
    "predict": main,
}


def app():
    Fire(torchyolo_app)


if __name__ == "__main__":
    app()
