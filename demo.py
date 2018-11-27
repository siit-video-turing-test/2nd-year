from __future__ import print_function, division
import sys
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import resnet

# usage: python demo.py input.mp4
# output: demo.avi

model = resnet.resnet50(num_classes=365, num_new_classes=19)
checkpoint = torch.load('lwf_best.pth.tar')
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model = model.cuda().eval()