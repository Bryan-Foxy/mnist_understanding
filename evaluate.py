#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:30:44 2024

@author: fozame
"""

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import arange

# Set theme
sns.set(style='whitegrid')

def evaluation(loss_1a, valid_1a, loss_1s, valid_1s, loss_2a, valid_2a, loss_2s, valid_2s, N, check_all=True):
    """
    Parameters
    ----------
    loss_1a : LIST
        The list of the loss train with training dnn with optimiser Adam.
    valid_1a : DICTIONARY
        The dictionnary(test_loss - accuracy) of the validation set with training dnn with optimiser Adam.
    loss_1s : LIST
        The list of the loss train with training dnn with optimiser SGD.
    valid_1s : DICTIONARY
        The dictionnary(test_loss - accuracy) of the validation set with training dnn with optimiser SGD.
    loss_2a : LIST
        The list of the loss train with training cnn with optimiser Adam.
    valid_2a : DICTIONARY
        The dictionnary(test_loss - accuracy) of the validation set with training dnn with optimiser Adam.
    loss_2s : LIST
        The list of the loss train with training cnn with optimiser SGD.
    valid_2s : DICTIONARY
        The dictionnary(test_loss - accuracy) of the validation set with training cnn with optimiser SGD.
    check_all : bool, optional
        Whether to plot all comparisons. The default is True.

    Returns
    -------
    None.

    """
    
    #extract values here
    #We start with Adam
    loss_test1a = valid_1a[0]['test_loss']
    acc_test1a = valid_1a[1]['accuracy']
    loss_test2a = valid_2a[0]['test_loss'] 
    acc_test2a = valid_2a[1]['accuracy']
    
    #SGD
    loss_test1s = valid_1s[0]['test_loss']
    acc_test1s = valid_1s[1]['accuracy']
    loss_test2s = valid_2s[0]['test_loss']
    acc_test2s = valid_2s[1]['accuracy']
    
    N = arange(0, N, 1)

    #Figure
    #DNN comparaison
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].set_title("Dense Neural Network Comparaison between Adam & SGD")
    ax[0].plot(N, acc_test1a, label="Acc Adam DNN", marker='o', linewidth=2)
    ax[0].plot(N, acc_test1s, label="Acc SGD DNN", marker='o', linewidth=2)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    fig.tight_layout()
    sns.despine(ax=ax[0], right=True, top=True)
    
    #CNN comparaison
    ax[1].set_title("Convolution Neural Network Comparaison between Adam & SGD")
    ax[1].plot(N, acc_test2a, label="Acc Adam CNN", marker='o', linewidth=2)
    ax[1].plot(N, acc_test2s, label="Acc SGD CNN", marker='o', linewidth=2)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    fig.tight_layout()
    sns.despine(ax=ax[1], right=True, top=True)
    
    fig.show()
    
    if check_all == True:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].set_title("Comparaison Accuracy between DNN and CNN")
        ax[0].plot(N, acc_test1a, label="Acc Adam DNN", marker='o', linewidth=2)
        ax[0].plot(N, acc_test1s, label="Acc SGD DNN", marker='o', linewidth=2)
        ax[0].plot(N, acc_test2a, label="Acc Adam CNN", marker='o', linewidth=2)
        ax[0].plot(N, acc_test2s, label="Acc SGD CNN", marker='o', linewidth=2)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
    
        sns.despine(ax=ax[0], right=True, top=True)
        
        #loss
        ax[1].set_title("Comparaison Loss between DNN and CNN")
        ax[1].plot(N, loss_test1a, label="Loss Adam DNN", marker='o', linewidth=2)
        ax[1].plot(N, loss_test1s, label="Loss SGD DNN", marker='o', linewidth=2)
        ax[1].plot(N, loss_test2a, label="Loss Adam CNN", marker='o', linewidth=2)
        ax[1].plot(N, loss_test2s, label="Loss SGD CNN", marker='o', linewidth=2)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        
        fig.tight_layout()
        sns.despine(ax=ax[1], right=True, top=True)
        
        fig.show()
        fig.savefig('loss comparaison')