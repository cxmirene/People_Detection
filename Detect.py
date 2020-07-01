# -*- coding: utf-8 -*-
import cv2
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.utils.misc import Timer

# net_type = 'mb3-ssd-lite'
class Detect():
    class_names = []
    predictor = ""

    def Init_Net(self, net_type):
        print("==========Init Net==========")
        if net_type=="mb2-ssd":
            model_path = "./models/mobilenet2-ssd.pth"
            label_path = "./models/voc-model-labels_mb2.txt"
            self.class_names = [name.strip() for name in open(label_path).readlines()]
            net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
            net.load(model_path)
            self.predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
        elif net_type=="mb3-large-ssd":
            model_path = "./models/mobilenet3-large-ssd.pth"
            label_path = "./models/voc-model-labels_mb3.txt"
            self.class_names = [name.strip() for name in open(label_path).readlines()]
            net = create_mobilenetv3_ssd_lite("Large", len(self.class_names), is_test=True)
            net.load(model_path)
            self.predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200)


    def detect(self, orig_image):
        boxes, labels, probs = self.predictor.predict(orig_image, 10, 0.4)

        result = []
        for i in range(boxes.size(0)):
            if not self.class_names[labels[i]] == "person":
                continue
            location = boxes[i, :]
            # cv2.rectangle(orig_image, (location[0], location[1]), (location[2], location[3]), (255, 255, 0), 4)
            result.append([location[0], location[1], location[2], location[3]])
            #label = f"""{voc_dataset.self.class_names[labels[i]]}: {probs[i]:.2f}"""
            #label = self.class_names[labels[i]]+":"+str(probs[i])
            # cv2.putText(orig_image, label,
            #             (location[0] + 20, location[1] + 40),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1,  # font scale
            #             (255, 0, 255),
            #             1)  # line type
        # path = "run_ssd_example_output.jpg"
        # cv2.imwrite(path, orig_image)
        print("The number of person: "+str(len(probs)))
        return result

def test():
    detector = Detect()
    detector.Init_Net()
    pic1 = cv2.imread("/home/cxm-irene/pytorch-ssd_/test.jpg")
    result = detector.detect(pic1)
    print(result)

# test()
