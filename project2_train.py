'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network we used

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
parser.add_argument('--output_path', default='./', type=str,
                    help='The path that stores the log files.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='When using this option, only run the test functions.')
args = parser.parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# the function to calculate the validation accuracy
def accur_net(net, dataloader):
    # initialise variables
    net = net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            # get the inputs
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            
            # forward process
            outputs, aux2, aux1 = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # add to total label size and correct outputs number
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return 100 * correct / total # calculate accuracy

# training process. 
def train_net(net, trainloader, valloader, criterion, optimizer, scheduler):
    # initialise variables
    val_accuracy = 0
    net = net.train()
    # use cuda if called with '--cuda'.use pretrained if called with '--pretrained'.
    if args.pretrained:
        if args.cuda:
            net.load_state_dict(torch.load(args.output_path + 'project2.pth', map_location='cuda'))
        else:
            net.load_state_dict(torch.load(args.output_path + 'project2.pth', map_location='cpu'))
            
    for epoch in range(1):  # loop over the dataset multiple times
        # initialise loss
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, aux2, aux1 = net(inputs)
            loss = criterion(outputs, labels) + 0.3*criterion(aux2, labels) + 0.3*criterion(aux1, labels) # important part: we add 0.3 * the auxiliary classifier loss
            loss.backward()
            optimizer.step()

            if args.cuda:
                loss = loss.cpu()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()
            
    # calculate validation accuracy
    val_accuracy = accur_net(net, valloader)

    # save network
#     torch.save(net.state_dict(), args.output_path + 'modified.pth')
    return val_accuracy

##############################################

############################################
# Transformation definition

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=.3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

####################################

####################################
# Define the training dataset and dataloader.

train_image_path = 'train/' 
validation_image_path = 'validation/' 

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                         shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=10,
                                         shuffle=True, num_workers=8)
####################################

# ==================================

network = Network()
if args.cuda:
    network = network.cuda()

# optimizer and scheduler
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
# train and eval trained network
if __name__ == '__main__':     # this is used for running in Windows
    val_acc = train_net(network, trainloader, valloader, criterion, optimizer, scheduler)
    print("final validation accuracy:%.2f%%" % (val_acc))

# ==================================
