<div align="center">
<h1>
  Yolov8 Architecture
</h1>
<h4>
    <img width="700" alt="teaser" src="https://github.com/kadirnar/torchyolo/blob/torchview/docs/yolox/yolox_s.gif">
</h4>
</div>

## Model Architecture
```python
from torchyolo import YoloHub

model = YoloHub(
  model_type="yolox", 
  model_path="yolox_s.pt", 
  config_path="yolox_s",
  device="cuda:0", 
  image_size=640)
result = model.view_model_architecture(file_format="pdf")
