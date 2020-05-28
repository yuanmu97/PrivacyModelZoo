from pmz import ObjectDetection
import sys 
import cv2 


if __name__ == "__main__":
    img_path = sys.argv[1]
    m = ObjectDetection(cfg_path="pmz/object/yolov3/config/yolov3.cfg", 
                        weights_path="pmz/object/yolov3/weights/yolov3.weights",
                        class_path="pmz/object/yolov3/data/coco.names")
    res = m.inference(img_path)
    print(res)
    out_img = cv2.imread(img_path)
    for r in res:
        startX, startY, endX, endY = r["box_points"]
        cv2.rectangle(out_img, (startX, startY), (endX, endY), (0,255,0), 2)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(out_img, r["name"], (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("res.jpg", out_img)
    