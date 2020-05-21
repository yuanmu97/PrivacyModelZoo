from pmz import SceneClassification
import sys 


if __name__ == "__main__":
    img_path = sys.argv[1]
    m = SceneClassification(model_path="./pmz/scene/data/resnet50_places365.pth.tar", 
                            json_path="./pmz/scene/data/model_class.json")
    m.loadFullModel()
    res = m.inference(img_path)
    print(res)