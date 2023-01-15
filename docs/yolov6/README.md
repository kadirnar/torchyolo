<div align="center">
<h1>
  Yolov6 Architecture
</h1>
<h4>
    <img width="700" alt="teaser" src="https://github.com/kadirnar/torchyolo/blob/torchview/docs/yolov6/yolov6n.gif">
</h4>
</div>

## Model Architecture
```python
from torchyolo import YoloHub

model = YoloHub(
  model_type="yolov6", 
  model_path="yolov6n.pt", 
  device="cuda:0", 
  image_size=640)
result = model.view_model(file_format="pdf")
