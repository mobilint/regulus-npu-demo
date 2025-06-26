import sys

import cv2
import numpy as np

import maccel

RESNET50_MXQ_PATH = "resnet50.mxq"
DATASET = [
    ("ILSVRC2012_val_00000001.JPEG", 65),
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_image(img, target_h, target_w):
    h, w = img.shape[:2]
    sh = (h - target_h) // 2
    sw = (w - target_w) // 2
    return img[sh : sh + target_h, sw : sw + target_w]


def preprocess_resnet50(img_path: str, mean=MEAN, std=STD):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = crop_image(img, 224, 224)
    img = img.astype(np.float32) / 255.0
    out = (img - mean) / std
    return out


def main(args):
    mxq_path = RESNET50_MXQ_PATH
    if len(args) == 2:
        mxq_path = args[1]

    acc = maccel.Accelerator()
    model = maccel.Model(mxq_path)
    model.launch(acc)
    for filename, label in DATASET:
        img = preprocess_resnet50(filename)
        result = model.infer([img])[0]

        print("label   :", label)
        print("predict :", np.array(result).argmax())

    model.dispose()


if __name__ == "__main__":
    main(sys.argv)
