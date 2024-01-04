# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:09:28 2023

@author: xphones
"""

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import NeuralNetwork, CNN
from train import train, test
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
        


#plot_images()


bs = 64 #batch size parameters

train_loader = DataLoader(training_data, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

#Display images
train_feature, train_label = next(iter(train_loader))
print("Feature batch size: {}".format(train_feature.size()))
print("Label batch size: {}".format(train_label.size()))
img = train_feature[0].squeeze()
label = train_label[0]
plt.title(label)
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


model = NeuralNetwork(input_f, h, output_f).to(device)
print(model)

loss_fn = torch.nn.CrossEntropyLoss() #loss
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
epochs = 5

#train
print("The model is starting: with Adam\n")
for t in range(epochs):
    print(f"Epoch {t+1}: -------------------------------")
    loss = train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)


print('Done')

fig , ax = plt.subplots()
ax.set_xlabel("Over Epochs")
ax.set_ylabel(f"Loss [Adam]")
ax.plot(loss)

print("The model is starting: with SGD\n")
opt_sgd = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
for t in range(epochs):
    print(f"Epoch {t+1}: -------------------------------")
    loss = train(train_loader, model, loss_fn, opt_sgd)
    test(test_loader, model, loss_fn)

print('Done')
fig , ax = plt.subplots()
ax.set_xlabel("Over Epochs")
ax.set_ylabel(f"Loss [SGD]")
ax.plot(loss)

#####CNN########################################
cnn = CNN(input_f, h, output_f).to(device)
print(cnn)
print("The CNN model start...")
for t in range(epochs):
    print(f"Epoch {t+1}: -------------------------------")
    loss = train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)



###################################################
#Nous avons retourner le nom, les poids, les biais#
#et l'erreur. mais nous avons des zones d'ombres.##
#car la taille de a = 6, b = 3, c = 3, d =10      #
#nous retenons que la taille de d = 10 represente #
#l'erreur. a,b,c contiennent des listes par contre#
# d non.                                          #
###################################################


#a,b,c,d = trained_params
#print(a)
#print("="*100)
#print(b)
#print("="*100)
#print(c)
#print("="*100)
#print(d)
#print("shape: {}".format(len(b[-1])))

#La taille de chque liste pour b,c est de 10. C'est excellent pour tracer un contourplot







                              
        


