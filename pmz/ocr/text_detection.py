import tesserocr
from PIL import Image


class TextDetection(object):
    
    def __init__(self):
        pass

    def inference(self, img_path):
        img = Image.open(img_path)
        res = [{
            "name": "text",
            "text": tesserocr.image_to_text(img)
        }]
        return res