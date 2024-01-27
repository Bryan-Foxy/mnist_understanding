# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:13:53 2023

@author: xphones

"""

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def trainer(dataloader, validloader, model, criterion, optimizer, epochs=5):
    """
    This function allows you to train a model using training data sets (dataloader) while evaluating its performance on validation data. 
    Using tqdm (taqadum) provides a visual progress bar to track training and assessment progress.
    A separate progress bar is used for each training epoch, showing progress across training batches, as well as a separate progress bar for evaluation on validation data.
    ---
    We return two lists, loss_cache and valid_cache, which respectively contain the average training losses for each epoch and the evaluation results on the validation data, 
    such as average loss and accuracy.
    """
    size = len(dataloader)
    model.train()
    loss_cache = []
    valid_cache = []
    
    #run tensorboard
    writer = SummaryWriter()
    step = 0

    for i in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        total_loss = 0

        for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training Batches", unit="batch", leave=False)):
            # compute prediction and loss
            preds = model(X)
            loss = criterion(preds, y)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / size
        loss_cache.append(avg_loss)
        print(f"\nEpoch {i + 1}/{epochs} - Avg Training Loss: {avg_loss:.6f}")

        # Evaluation on validation set
        model.eval()
        total_correct = 0
        total_test_loss = 0

        with torch.no_grad():
            for X_val, y_val in tqdm(validloader, desc="Validation Batches", unit="batch", leave=False):
                preds_val = model(X_val)
                total_test_loss += criterion(preds_val, y_val).item()
                total_correct += (preds_val.argmax(1) == y_val).type(torch.float).sum().item()

        avg_test_loss = total_test_loss / len(validloader)
        accuracy = total_correct / len(validloader.dataset)
        valid_cache.append({"test_loss": avg_test_loss, "accuracy": accuracy})
        
        #add plot in the tensorboard
        writer.add_scalar("Loss", loss, global_step=step)
        writer.add_scalar("Accuracy", accuracy, global_step=step)
        step += 1
        
        #print loss and accuracy for the valid loader
        print(f"Validation - Avg Loss: {avg_test_loss:.6f} | Accuracy: {accuracy * 100:.2f}%\n")

    return loss_cache, valid_cache