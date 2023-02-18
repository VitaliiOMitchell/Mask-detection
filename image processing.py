import cv2 as cv
import os
import pickle
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes

class Image_prep:
    def __init__(self, path):
        self.path = path

    def draw_boxes(self, data):
        for image in os.listdir(self.path):
            img = cv.imread(os.path.join(self.path, image))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            

if __name__ == '__main__':
    path = 'F:/Python/Projects/Mask-detection/images_masks'
    with open('data_for_detection.pkl', 'rb') as d:
        data = pickle.load(d)
    