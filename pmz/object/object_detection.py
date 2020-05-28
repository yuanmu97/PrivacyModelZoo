from __future__ import division
from .yolov3.models import *
from .yolov3.utils.utils import *
from .yolov3.utils.datasets import *
from PIL import Image
import torch
from torch.autograd import Variable

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ObjectDetection(object):

    def __init__(self, cfg_path, weights_path, class_path, img_size=416):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(cfg_path, img_size=img_size).to(self.device)
        self.img_size = img_size

        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weights_path))

        self.model.eval()  # Set in evaluation mode
        self.classes = load_classes(class_path)  # Extracts class labels from file

    def inference(self, img_path):
        img = Image.open(img_path)
        inp_img = transforms.ToTensor()(img)
        # Pad to square resolution
        inp_img, _ = pad_to_square(inp_img, 0)
        # Resize
        inp_img = resize(inp_img, self.img_size)
        inp_img = Variable(inp_img.unsqueeze(0))
        np_img = np.array(img)
        with torch.no_grad():
            detections = self.model(inp_img)
            detections = non_max_suppression(detections, 0.8, 0.4)
        res = []
        for detection in detections:
            detection = rescale_boxes(detection, self.img_size, np_img.shape[:2])
            unique_labels = detection[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))
                # print(x1,y1,x2,y2)
                res.append({
                    "name": self.classes[int(cls_pred)],
                    "score": cls_conf.item(),
                    "box_points": [int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())]
                })
        return res