# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:13:53 2023

@author: xphones
"""

import torch
 

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        #compute prediction and loss
        preds = model(X)
        loss = loss_fn(preds, y)
        
        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch%100 == 0:
            loss, current = loss.item(), batch+1 * len(X)
            print(f"Loss: {loss:>7f}   [{current:>5d}|{size:>5d}]")
            

            

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    
    
    with torch.no_grad():
        for X,y in dataloader:
            preds = model(X)
            test_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1)==y).type(torch.float).sum().item()
            
        test_loss /=  num_batches
        correct /= size
        print(f"Test Error: \nAccuracy {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")
    
    
        
        
    
    
    