import os
import json
import cv2
import torch
from torchvision.datasets import VOCDetection
from pprint import pprint
import numpy as np
from torchvision.transforms import Compose, Resize, RandomAffine, ColorJitter, ToTensor, Normalize
from torch.utils.data import Dataset
from PIL import Image


# volleyball dataset
class VolleyballDataset(Dataset):
    def __init__(self, transform=None, mode = 'Train'):
        if transform is None:
            self.transform = Compose([])
        else:
            self.transform = transform


        with open("Data/dataset/volleyball/classes.txt") as f:
            labels = f.readlines()
        self.labels = [label.strip() for label in labels]
        self.image = []
        for file in os.listdir("Data/dataset/volleyball/"):
            if file.endswith(".jpg"):
                self.image.append("Data/dataset/volleyball/" + file)
        l = len(self.image)
        if mode == 'Train':
            self.image = self.image[:int(0.8*l)]
        else:
            self.image = self.image[int(0.8*l):]
        self.categories = ['Player', 'Referee', 'Ball']

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = Image.open(self.image[idx])
        old_h, old_w = image.size
        image = self.transform(image)
        _, new_h, new_w = image.shape
        targets = {}
        file = self.image[idx].replace(".jpg", ".txt")
        with open(file) as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        annotation = []
        labels = []
        for line in lines:
            annotation.append([float(i) for i in line.split(' ')])
        for anno in annotation:
            labels.append(int(anno[0]))
            anno.pop(0)
        boxes = []
        for anno in annotation:
            x_center, y_center = anno[0], anno[1]
            width, height = anno[2], anno[3]
            x_min = (x_center - width / 2) * new_w
            y_min = (y_center - height / 2) * new_h
            x_max = (x_center + width / 2) * new_w
            y_max = (y_center + height / 2) * new_h
            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        targets["boxes"] = boxes
        targets["labels"] = labels
        return image, targets



# volleyball dataset
class FootballDataset(Dataset):
    def __init__(self, transform=None, mode = 'Train'):
        if transform is None:
            self.transform = Compose([])
        else:
            self.transform = transform


        with open("Data/dataset/football/classes.txt") as f:
            labels = f.readlines()
        self.labels = [label.strip() for label in labels]
        self.image = []
        for file in os.listdir("Data/dataset/football/"):
            if file.endswith(".jpeg"):
                self.image.append("Data/dataset/football/" + file)
        l = len(self.image)
        if mode == 'Train':
            self.image = self.image[:int(0.8*l)]
        else:
            self.image = self.image[int(0.8*l):]
        self.categories = ['Ball', 'Player']

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = Image.open(self.image[idx])
        old_h, old_w = image.size
        image = self.transform(image)
        _, new_h, new_w = image.shape
        targets = {}
        file = self.image[idx].replace(".jpeg", ".txt")
        with open(file) as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        annotation = []
        labels = []
        for line in lines:
            annotation.append([float(i) for i in line.split(' ')])
        for anno in annotation:
            labels.append(int(anno[0]))
            anno.pop(0)
        boxes = []
        for anno in annotation:
            x_center, y_center = anno[0], anno[1]
            width, height = anno[2], anno[3]
            x_min = (x_center - width / 2) * new_w
            y_min = (y_center - height / 2) * new_h
            x_max = (x_center + width / 2) * new_w
            y_max = (y_center + height / 2) * new_h
            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        targets["boxes"] = boxes
        targets["labels"] = labels
        return image, targets



#pascal voc 2007
class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download,transform):
        super(VOCDataset, self).__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        old_height, old_width = float(annotations['annotation']['size']['height']), float(annotations['annotation']['size']['width'])
        new_height, new_width = image.shape[1:]
        labels = []
        bboxes = []
        for obj in annotations['annotation']['object']:
            lst = [int(i) for i in obj['bndbox'].values()]
            lst[0] = lst[0] / old_width * new_width
            lst[1] = lst[1] / old_height * new_height
            lst[2] = lst[2] / old_width * new_width
            lst[3] = lst[3] / old_height * new_height
            bboxes.append(lst)
            labels.append(self.categories.index(obj['name']))
        boxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        targets = {}
        targets['boxes'] = boxes
        targets['labels'] = labels



        return image, targets

if __name__ == '__main__':
    idx = 1
    train_transform = Compose([
        # Resize((416, 416)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    data = FootballDataset(train_transform, mode='Test')
    imgg, target = data[idx]
    cv2.imwrite('Data/test.jpeg', cv2.cvtColor(np.array(imgg).transpose((1,2,0)) * 255., cv2.COLOR_RGB2BGR))


