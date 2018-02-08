# TODO: Create comment block explaining the def
# Trainer for the Pokemon Dataset
# Trains with your choice of optimizer, model and train loader
# Uses Cross Entropy Loss for the loss function
# Returns the losses and the fitted model



import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def train(epochs, train_loader, model, optimizer):
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        len_train_loader = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = torch.nn.CrossEntropyLoss()
            len_train_loader = len(train_loader.dataset)
            # Always have requires_grad set to false since we don't need gradients.
            data, target = Variable(data, requires_grad=False), Variable(target.type(torch.LongTensor), requires_grad=False)
            optimizer.zero_grad()
            output = model(data)

            # This is used to be able to show our accuracy it finds the max output, one with the higher probability
            _, predicted = torch.max(output.data, 1)

            # Calculating our loss
            output_loss = loss(output, target)
            output_loss.backward()

            # Stepping
            optimizer.step()

            correct = (predicted == target.data).sum()
            # Set equal to batch size to help figure out accuracy
            total = 20
            total_loss = total_loss + output_loss.data[0]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), output_loss.data[0]/total))
            print('Accuracy of the network: %d %%' % (
                100 * correct / total))
        losses.append(total_loss/len_train_loader)
    return losses, model
