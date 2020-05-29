from pmz import TextDetection
import sys 


if __name__ == "__main__":
    img_path = sys.argv[1]
    m = TextDetection()
    res = m.inference(img_path)
    print(res)