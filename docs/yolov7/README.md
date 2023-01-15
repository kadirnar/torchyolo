<div align="center">
<h1>
  Yolov7 Architecture
</h1>
<h4>
    <img width="700" alt="teaser" src="https://github.com/kadirnar/torchyolo/blob/torchview/docs/yolov7/yolov7_tiny.gif">
</h4>
</div>

## Model Architecture
```python
from torchyolo import YoloHub

model = YoloHub(
  model_type="yolov7", 
  model_path="yolov7.pt", 
  device="cuda:0", 
  image_size=640)
result = model.view_model(file_format="pdf")
