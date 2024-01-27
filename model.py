# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:44:18 2023

@author: xphones
"""

import torch
import torch.nn as nn

torch.manual_seed(123)

class NeuralNetwork(nn.Module):
    def __init__(self, input_f, h, output_f):
        super(NeuralNetwork,self).__init__()
        self.input_f = input_f
        self.h = h
        self.output_f = output_f
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(input_f, h),
            nn.ReLU(),
            nn.Linear(h,h),
            nn.ReLU(),
            nn.Linear(h,output_f),
            
            )
        
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        
        return x
    



class CNN(nn.Module):
    def __init__(self, input_h, h, output_h):
        super(CNN, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,8,5),
            nn.Conv2d(8,12,5)
        )

        self.dnn = nn.Sequential(
            nn.Linear(12*4*4, h),
            nn.ReLU(),
            nn.Linear(h,h),
            nn.ReLU(),
            nn.Linear(h,output_h),
        ) 

    def forward(self, x):
        x = self.convolution(x)
        x = torch.flatten(x,1)
        x = self.dnn(x)
        
        return x
        


    
        
        
        
