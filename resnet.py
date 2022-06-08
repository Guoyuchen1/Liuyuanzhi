import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    img = cv2.imread('data_set/train_pic/train_0.bmp')
    cv2.imshow('img',img)
    cv2.waitKey(0)
    img = transform(img)
    print(img)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)

    img = img.to(device)

    net = models.resnet101().to(device)
    net.load_state_dict(torch.load('resnet101-5d3b4d8f.pth'))
    exact_list = ['layer4']
    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    #提取特征
    print(outs['layer4'].size())
    feature=outs['layer4']

if __name__ == '__main__':
    get_feature()
