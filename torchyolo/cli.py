from fire import Fire
from torchyolo import YoloHub
from torchyolo import __version__ as torchyolo_version


torchyolo_app = {
    "version": torchyolo_version,
    "hub": YoloHub,
}

def app():
    Fire(torchyolo_app)
    
if __name__ == "__main__":
    app()
