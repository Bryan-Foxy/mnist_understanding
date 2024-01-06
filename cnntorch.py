"""
Created on Mon May  1 17:04:47 2023

@author: armandbryan
"""

#import basic librairies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device config
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
num_epochs=20
batch_size=4
learning_rate=0.001


#Dowload the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=transform)
print('Download train data successful')

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=transform)
print('Download test data successful')
print("Shape: ", len(train_dataset))

labels_map = {
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

img, lab = train_dataset
plt.imshow(img[1])
sfsd


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               shuffle=False)
print(len(train_dataloader))
for i, batch in enumerate(train_dataloader):
    images, labels = batch
    print("images shape: ", images.shape)

    # Plot images
    for j in range(4):
        plt.subplot(1, batch_size, j + 1)
        plt.imshow(images[j].permute(1, 2, 0))  # Permute channels for proper display
        plt.title(f"Label: {labels[j]}")
        plt.axis("off")

    plt.show()

    if i == 1:
        break

print('Convert to DataLoader finish...')


#implement convnet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #initialize features
        self.gradient = None
        self.reisze = torchvision.transforms.Resize(28)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        self.conv3 = nn.Conv2d(8, 16, 5)
        self.conv4 = nn.Conv2d(16, 20, 5)
        self.fc1 = nn.Linear(20*5*5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 =nn.Linear(256, 10)

    
    def get_activations(self, x):
        with torch.no_grad():
            return self.conv4(x)
    
    def hook_activation(self, grad):
        self.gradient = grad
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.pool(F.relu(self.conv4(x)))
        h = x.register_hook(self.hook_activation)
        x=x.view(-1, 20*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
model = ConvNet().to(device)
print(model)
m = model.conv4
print(m)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        
        images = images.to(device)
        labels= labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(i+1) % 2000==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

with torch.no_grad():
    n_correct=0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in test_dataloader:
        images=images.to(device)
        labels = labels.to(device)
        outputs=model(images)
        #max returns
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if(label==pred):
                n_class_correct[label]+=1
            n_class_samples[label] +=1
    
    acc = 100.0 *n_correct/n_samples
    print(f'Accuracy of the network: {acc} %')
    
    for i in range(10):
        acc=100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

#let's do grad cam to visualize output of the model

def get_grad_cam(model, image, labels, size):
    """
    This functions helps to visualize how CNN capture information
    """
    out = model(image)
    print(out[:,label])
    out[:,label].backward() #perfom partial derivative
    activation = model.get_activations(image)
    weights = torch.sum(model.gradient, axis = [0,2,3])
    scalar_prod = torch.tensordot(weights, activation,
                                  dims = ([0], [1]))
    scalar_prod = model.relu(scalar_prod)
    scalar_prod /= torch.max(scalar_prod)

    return model.resize(scalar_prod)

    
    
    