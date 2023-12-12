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
        super().__init__()
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
        
        
        
