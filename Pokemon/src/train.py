# TODO: Create comment block explaining the def




import torch
from torch.autograd import Variable


# TODO: Write a way to plot losses
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
            _, predicted = torch.max(output.data, 1)
            output_loss = loss(output, target[:, 0])
            output_loss.backward()
            optimizer.step()
            total = 56
            correct = (predicted == target[:, 0].data).sum()
            total_loss = total_loss + output_loss.data[0]
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), output_loss.data[0]))
                print('Accuracy of the network: %d %%' % (
                    100 * correct / total))
        losses.append(total_loss/len_train_loader)
    return losses, model
