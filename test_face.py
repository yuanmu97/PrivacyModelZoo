from pmz import FaceDetection
import sys 


if __name__ == "__main__":
    img_path = sys.argv[1]
    m = FaceDetection()
    res = m.inference(img_path, output_image_path="res.jpg", threshold=0.)
    print(res)