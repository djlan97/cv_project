import os

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import datasets, models


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = 4

    data_dir = 'data/dataset'

    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                   data_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                             shuffle=True, num_workers=6)

    classes = testset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # Make a grid from batch
    out = torchvision.utils.make_grid(images)

    imshow(out, title=[classes[x] for x in labels])
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

    net = models.resnet34()

    num_ftrs = net.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    net.fc = nn.Linear(num_ftrs, len(classes))

    net = net.to(device)
    images = images.to(device)

    net.load_state_dict(torch.load('results/resnet34/resnet34_checkpoint.pt'))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                              for j in range(len(labels))))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the testset: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')