<div align="center">
<h1>
  TorchYolo: This is an open source toolbox for YOLO series algorithms based on PyTorch.
</h1>
</div>


### Introduction

The TorchYolo library aims to support all YOLO models(like YOLOv5, YOLOv6, YOLOv7, YOLOX, etc.) and provide a unified interface for training and inference. The library is based on PyTorch and is designed to be easy to use and extend.

### Installation
```bash
git clone https://github.com/kadirnar/torchyolo
cd torchyolo
pip install -r requirements.txt
```
or 
```bash
pip install torchyolo
```
### Usage
```python
python torchyolo/predict.py --config configs/default_config.yaml
```
Note: You only need to make changes in the default_config.yaml file.

# Contributing
Before opening a PR:
  - Install required development packages:
    ```bash
    pip install -r requirements-dev.txt
    ```
  - Reformat the code with black and isort:
    ```bash
    bash script/code_format.sh
    ``` 

### TODO
- [ ] Add more models(YOLOV4, Scaled-YOLOv4, YOLOR)
- [ ] Add tracker algorithm(Sort, StrongSort, ByteTrack, OcSort, etc.)
- [ ] Add Train, Export and Eval scripts
- [ ] Add Benchmark Results
