# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:09:28 2023

@author: xphones
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import NeuralNetwork, CNN
from train import trainer
from evaluate import evaluation

import matplotlib.pyplot as plt

print('version of pytorch {}'.format(torch.__version__))

training_data = datasets.MNIST(root='data',
                               train = True,
                               download= True,
                               transform = ToTensor()
                               )

test_data = datasets.MNIST(root='data',
                               train = False,
                               download= True,
                               transform = ToTensor()
                               )


labels_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
    }

def plot_images(train_data = training_data, label = labels_map, xsize=8, ysize=8, col=3, row=3):
    """
    

    Parameters
    ----------
    train_data : Torch.datasets convert in tensor
        DESCRIPTION. This variable contains training images.
    label : Dictionnary
        DESCRIPTION. Label of training images.
    xsize : Integer
        DESCRIPTION. Width of the figure.
    ysize : Integer
        DESCRIPTION. Height of the figure.
    col : Integer
        DESCRIPTION. Colums for subplots.
    row : Integer
        DESCRIPTION. Rows for subplots.

    Returns
    -------
    None.

    """
    
    figure = plt.figure(figsize=(xsize,ysize))
    cols, rows = col, row
    for i in range(1, cols*rows +1):
        
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()
        


plot_images()


bs = 64 #batch size parameters

train_loader = DataLoader(training_data, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
print("We have {} train batches".format(len(train_loader)))
print("We have {} test batches".format(len(test_loader)))

#Display images
train_feature, train_label = next(iter(train_loader))
print("Feature batch size: {}".format(train_feature.size()))
print("Label batch size: {}".format(train_label.size()))
idx = torch.randint(low=0, high=64, size=(1,))
img = train_feature[idx].squeeze()
label = train_label[idx]
plt.title("id {} | label {}".format(idx,label))
plt.imshow(img, cmap='gray')
plt.show()


#Select the device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )

print(f"We are using {device} device for this project")


#hyperparams
input_f = 28*28
h = 512
output_f = 10


dnn = NeuralNetwork(input_f, h, output_f).to(device)
print(dnn)

loss_fn = torch.nn.CrossEntropyLoss() #loss
lr = 1e-3
opt_1a = torch.optim.Adam(dnn.parameters(), lr = lr) 
opt_1s = torch.optim.SGD(dnn.parameters(), lr = lr)
opt_1 = [opt_1a,opt_1s]
epochs = 5


#####DNN########################################
print("The model is starting: with Adam\n")
print(dnn)
total_params_dnn = sum(param.numel() for param in dnn.parameters() if param.requires_grad)
print("{} parameters trainables".format(total_params_dnn))
print("The DNN model start...")
for opt in opt_1:
    if opt == opt_1a:
        print("With Adam: \n")
        loss_1a, valid_1a = trainer(train_loader, test_loader, dnn, loss_fn, opt, epochs=epochs)
    else:
        print("With SGD: \n")
        loss_1s, valid_1s = trainer(train_loader, test_loader, dnn, loss_fn, opt, epochs=epochs)
        
#####CNN########################################
cnn = CNN(input_f, h, output_f).to(device)
opt_2a = torch.optim.Adam(cnn.parameters(), lr = lr)
opt_2s = torch.optim.SGD(cnn.parameters(), lr = lr)
opt_2 = [opt_2a, opt_2s]
print(cnn)
total_params_cnn = sum(param.numel() for param in cnn.parameters() if param.requires_grad)
print("{} parameters trainables".format(total_params_cnn))
print("The CNN model start...")
for opt in opt_2:
    if opt == opt_2a:
        print("With Adam: \n")
        loss_2a, valid_2a = trainer(train_loader, test_loader, cnn, loss_fn, opt, epochs=epochs)
    else:
        print("With SGD: \n")
        loss_2s, valid_2s = trainer(train_loader, test_loader, cnn, loss_fn, opt, epochs=epochs)
        
###Evaluation####################################
evaluation(loss_1a, valid_1a, loss_1s, valid_1s, loss_2a, valid_2a, loss_2s, valid_2s, N=epochs)