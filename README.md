<div align="center">
<h2>
  TorchYolo: YOLO Series Object Detection and Track Algorithm Library
</h2>
<h4>
    <img width="700" alt="teaser" src="https://github.com/kadirnar/TrackerHub/blob/main/docs/norfair_track/norfair_demo.gif">
<div>
    <a href="https://pepy.tech/project/torchyolo"><img src="https://pepy.tech/badge/torchyolo" alt="downloads"></a>
    <a href="https://badge.fury.io/py/torchyolo"><img src="https://badge.fury.io/py/torchyolo.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/kadirnar/torchyolo"><img src="https://img.shields.io/badge/%20HuggingFace%20-Demo-blue.svg" alt="HuggingFace Spaces"></a>
</div>
</div>


### Introduction

The TorchYolo library aims to support YOLO models(like YOLOv5, YOLOv6, YOLOv7, YOLOv8) and Tracker Algorithm(Sort, StrongSort, ByteTrack, OcSort and Norfair) and provide a unified interface for training and inference. The library is based on PyTorch and is designed to be easy to use and extend.

### Installation 
```bash
pip install torchyolo
```
### Usage
First download the [default_config.yaml](https://github.com/kadirnar/torchyolo/releases/download/v1.0.0/default_config.yaml) file.

```python
from torchyolo import YoloHub

model = YoloHub(
    config_path="default_config.yaml",
    model_type="yolov8",
    model_path="yolov8s.pt",
)
result = model.predict(
    source="test.mp4", 
    tracker_type="NORFAIR", # or False
    tracker_config_path="norfair_track.yaml"
)
```

### Detect Configuration
```yaml
DETECTOR_CONFIG:
  # The threshold for the IOU score
  IOU_TH: 0.45
  # The threshold for the confidence score
  CONF_TH: 0.25
  # The size of the image
  IMAGE_SIZE: 640
  # The device to run the detector
  DEVICE: cuda:0
  # F16 precision
  HALF: False
  # The path of the yolov6 label file
  YOLOV6_YAML_FILE: torchyolo/configs/yolov6/coco.yaml
  # The path of the yolovx config file
  YOLOX_CONFIG_PATH: configs.yolox.yolox_s
  # The path of the Hugging Face model
  HUGGING_FACE_MODEL: False 

DATA_CONFIG:
  # The path of the output video
  OUTPUT_PATH: output.mp4
  # Save the video
  SHOW: False 
  # Show the video
  SAVE: True

```

### Tracker Config File

ByteTrack: https://github.com/kadirnar/torchyolo/releases/download/v0.0.5/default_config.yaml

OcSort: https://github.com/kadirnar/torchyolo/releases/download/v0.0.5/oc_sort.yaml

StrongSort: https://github.com/kadirnar/torchyolo/releases/download/v0.0.5/strong_sort.yaml

Norfair: https://github.com/kadirnar/torchyolo/releases/download/v0.0.5/norfair_track.yaml

Sort: https://github.com/kadirnar/torchyolo/releases/download/v0.0.5/sort_track.yaml

## Model Architecture
```python
from torchyolo import YoloHub

model = YoloHub(config_path="torchyolo/default_config.yaml")
result = model.view_model(file_format="pdf")
```

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

### Acknowledgement
A part of the code is borrowed from [SAHI](https://github.com/obss/sahi). Many thanks for their wonderful works.

### Citation
```bibtex
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
```bibtex
@article{li2022yolov6,
  title={YOLOv6: A single-stage object detection framework for industrial applications},
  author={Li, Chuyi and Li, Lulu and Jiang, Hongliang and Weng, Kaiheng and Geng, Yifei and Li, Liang and Ke, Zaidan and Li, Qingyuan and Cheng, Meng and Nie, Weiqiang and others},
  journal={arXiv preprint arXiv:2209.02976},
  year={2022}
}
```
```bibtex
@software{glenn_jocher_2020_4154370,
  author       = {Glenn Jocher and,Alex Stoken and,Jirka Borovec and,NanoCode012 and,ChristopherSTAN and,Liu Changyu and,Laughing and,tkianai and,Adam Hogan and,lorenzomammana and,yxNONG and,AlexWang1900 and,Laurentiu Diaconu and,Marc and,wanghaoyang0106 and,ml5ah and,Doug and,Francisco Ingham and,Frederik and,Guilhen and,Hatovix and,Jake Poznanski and,Jiacong Fang and,Lijun Yu 于力军 and,changyu98 and,Mingyu Wang and,Naman Gupta and,Osama Akhtar and,PetrDvoracek and,Prashant Rai},
  title={{ultralytics/yolov5: v7.2 - Bug Fixes and 
                   Performance Improvements}},
  month= oct,
  year= 2020,
  publisher= {Zenodo},
  version= {v3.1},
  doi= {10.5281/zenodo.4154370},
  url= {https://doi.org/10.5281/zenodo.4154370}
}
```
```bibtex
@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
```bibtex
@article{du2022strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Song, Yang and Yang, Bo and Zhao, Yanyun},
  journal={arXiv preprint arXiv:2202.13514},
  year={2022}
}
```
```bibtex
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```
```bibtex

@article{torchreid,
    title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
    author={Zhou, Kaiyang and Xiang, Tao},
    journal={arXiv preprint arXiv:1910.10093},
    year={2019}
} 
```
```bibtex
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```
```bibtex
@inproceedings{zhou2019osnet,
    title={Omni-Scale Feature Learning for Person Re-Identification},
    author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
    booktitle={ICCV},
    year={2019}
}
```
```bibtex
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```
```bibtex
@article{zhou2021osnet,
    title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
    author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
    journal={TPAMI},
    year={2021}
}
```
