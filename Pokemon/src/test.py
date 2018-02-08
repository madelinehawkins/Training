# Madeline Hawkins

import torch
from torch.autograd import Variable



def test(model, data_loader):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    losses = []
    for data, target in data_loader:
        data, target = Variable(data, requires_grad=False), Variable(target.type(torch.LongTensor), requires_grad=False)
        output = model(data)
        test_loss += loss(output, target).data[0]
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target.data).sum()
        losses = losses + [(predicted == target.data).sum()]
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
             test_loss, correct, len(data_loader.dataset),
             100. * correct / len(data_loader.dataset)))
    print('\nTest Set: Accuracy across all classes: ')
    for x in losses:
        print('({:.0f}%)\n'.format(100. * x / 30))