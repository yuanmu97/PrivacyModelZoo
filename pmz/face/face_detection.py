import dlib
import cv2


class FaceDetection(object):
    """FaceDetection
    dlib face detection model
    """
    def __init__(self):
        self.model = dlib.get_frontal_face_detector()


    def inference(self, img_path, threshold=0.5, output_image_path=None):
        img = dlib.load_rgb_image(img_path)
        # run(img, upsample_rate, threshold)
        dets, scores, idx = self.model.run(img, 1, threshold)
        results = []
        if output_image_path:
            out_img = cv2.imread(img_path)
        for i,det in enumerate(dets):
            # print("Detection {}, score {}, face_type {}".format(
            #     det, scores[i], idx[i]
            # ))
            startX, endX, startY, endY = det.left(), det.right(), det.top(), det.bottom()
            results.append({"name": "face", 
                            "box_points": [startX, startY, endX, endY],
                            "score": scores[i],
                            "face_type": idx[i]})
            if output_image_path:
                cv2.rectangle(out_img, (startX, startY), (endX, endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(out_img, "face", (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if output_image_path:
            cv2.imwrite(output_image_path, out_img)
        return results
