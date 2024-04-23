#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:47:20 2024

@author: fozame
"""
#Load librairies
import numpy as np
import cv2
import torch

from torchvision import transforms
from torch.nn import functional as F
from torch import topk
from model import CNN
from main import test_loader

#Select the device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )

#load the model 
model = CNN()
model = model.eval()
path = 'model/mnist_cnn.pth'
model.load_state_dict(torch.load(path)) #here he load the weights and biais pretrained 


def return_cam(convs, w_softmax, class_idx):
    """
    

    Parameters
    ----------
    convs : torch type
        Represent the feature of the convolution layer.
    w_softmax : torch.float
        Represent the different weight for the output for the model by using softmax
    class_idx : integer
        Different classes id.

    Returns
    -------
    output_cam .
    
    Methods
    -------
    This function generate the class activation upsample to 256 * 256

    """
    
    size_upsample = (256, 256)
    bz, nc, h, w = convs.shape
    output_cam = []
    
    for idx in class_idx:
        cam = w_softmax[idx].dot(convs.reshape((nc, h*w)))
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        
        
    return output_cam


def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    """
    

    Parameters
    ----------
    CAMs : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    height : TYPE
        DESCRIPTION.
    orig_image : TYPE
        DESCRIPTION.
    class_idx : TYPE
        DESCRIPTION.
    save_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)))
        result = heatmap * 0.5 + orig_image * 0.5
        cv2.putText(result, str(int(class_idx[i])), (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('CAM', result/255.)
        cv2.waitKey(0)
        
def hook_feature(module, in_data, out_data):
    features_blobs.append(out_data.data.cpu().numpy())
    
    

features_blobs = []
model._modules.get('conv').register_forward_hook(hook_feature)

#get softmax weight
params= list(model.parameters())
w_softmax = np.squeeze(params[-2].data.numpy())

#transform method
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((28,28)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean = [.5], 
         std = [.5])]
    )

image_path = 'img/cam/'

for i,image in len(test_loader[5]):
    img = cv2.imread(image)
    orig_image = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=2)
    height, width, _ = orig_image.shape
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    outputs = model(img_tensor)
    probs = F.softmax(outputs).data.squeeze()
    class_idx = topk(probs, 1)[1].int()
    
    CAMs = return_cam(features_blobs[0], w_softmax, class_idx)
    save_name = f"{image_path.split('/')[-1].split('.')[0]}"
    show_cam(CAMs, width, height, orig_image, class_idx, save_name)
        






