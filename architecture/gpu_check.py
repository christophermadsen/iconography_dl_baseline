import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == '__main__':
    model = models.resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    batch_size = 32

    dataset = datasets.FakeData(
        size=1000,
        transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        num_workers=2,
        pin_memory=True,
        batch_size = batch_size
    )

    model.to('cuda')

    dn = len(dataset)
    num_epochs = 15

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        count = 0
        running_loss = 0.0
        running_corrects = 0
        for data, target in loader:
            count += 1
            data = data.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f'{batch_size*count}/{dn}', end='\r')

            # statistics
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data)

        epoch_loss = running_loss / dn
        epoch_acc = running_corrects.double() / dn

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print('Done')
