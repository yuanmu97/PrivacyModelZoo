# PrivacyModelZoo

The ModelZoo for Hi-Privacy project. 


## Dependency

Face detection:

```bash
pip install opencv-python
pip install dlib
```

OCR:

https://github.com/sirfz/tesserocr

```bash
sudo apt-get install tesseract-ocr
```

Scene classification & object detection:


```bash
pip install torch torchvision
```

## APIs

```python
from pmz import FaceDetection, ObjectDetection, SceneClassification, TextDetection

img_path = "path/to/image"
# face
m1 = FaceDetection()
res = m.inference(img_path, output_image_path="res.jpg", threshold=0.)
"""
res = [
    {
        "name": "face",
        "score": confidence,
        "box_points": [x1, y1, x2, y2],
        "face_type": INT
    }, ...
]
"""
# object
m2 = ObjectDetection(cfg_path="pmz/object/yolov3/config/yolov3.cfg", 
                     weights_path="pmz/object/yolov3/weights/yolov3.weights",
                     class_path="pmz/object/yolov3/data/coco.names")
res = m2.inference(img_path)
"""
res = [
    {
        "name": object_label,
        "score": confidence,
        "box_points": [x1, y1, x2, y2]
    }, ...
]
"""
# scene
m3 = SceneClassification(model_path="./pmz/scene/data/resnet50_places365.pth.tar", 
                         json_path="./pmz/scene/data/model_class.json")
m3.loadFullModel()
res = m3.inference(img_path)
"""
res = [
    {
        "name": scene_label,
        "score": confidence
    }, ...
]
"""
# text
m4 = TextDetection()
res = m4.inference(img_path)
"""
res = [
    {
        "name": "text",
        "text": "detected_text"
    }
]
"""
```