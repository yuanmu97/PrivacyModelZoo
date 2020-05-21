import torchvision.models as tvmodels
import torch
from torch.nn import functional as F
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn
import json


class SceneClassification(object):
    def __init__(self, model_path, json_path):		
        self.model = tvmodels.__dict__['resnet50'](num_classes=365)
        self.centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model_path = model_path
        self.json_path = json_path
        self.classes = json.load(open(self.json_path, "r"))


    def loadFullModel(self):
        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    
    def inference(self, image_input, result_count=5):
        img = Image.open(image_input)
        img = V(self.centre_crop(img).unsqueeze(0))
        if torch.cuda.is_available():
            img = img.cuda()

        res = []
        try:
            logit = self.model.forward(img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs_tmp, idx = h_x.sort(0, True)

            for i in range(result_count):
                res.append({
                    "name": self.classes[str(idx[i].item())],
                    "score": probs_tmp[i].item()
                })

        except RuntimeError:
            print("Bad input data.")

        return res
