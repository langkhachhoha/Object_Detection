from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomAffine, ColorJitter, ToTensor, Normalize
from datasets import *
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2
from PIL import Image
from torchvision.ops import nms

model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
link = 'test.jpg'
cv2.imwrite(link, cv2.imread('Data/dataset/voc/VOCdevkit/VOC2007/JPEGImages/000019.jpg'))


img_cv = cv2.imread(link)
img = Image.open(link)
val_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])
imgg = [val_transform(img)]
model.eval()
predictions = model(imgg)
for prediction in predictions:
    anno = prediction['boxes']
    print(prediction)
    actual_anno = nms(anno, prediction['scores'], iou_threshold=0.4)
    print(actual_anno)
    for idx in actual_anno:
        x1, y1, x2, y2 = anno[idx]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('img.jpg', img_cv)
cv2.waitKey(0)










