import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import imshow

from serengeti_dataset import SerengetiDataset

if __name__ == '__main__':
    '''training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }'''

    training_data = SerengetiDataset('data/dataset/train.csv', 'data/dataset/train', transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    test_data = SerengetiDataset('data/dataset/test.csv', 'data/dataset/test', transforms.RandomResizedCrop(224))

    labels_map = {0: 'baboon',
                  1: 'buffalo',
                  2: 'cheetah',
                  3: 'eland',
                  4: 'elephant',
                  5: 'empty',
                  6: 'gazellegrants',
                  7: 'gazellethomsons',
                  8: 'giraffe',
                  9: 'guineafowl',
                  10: 'hartebeest',
                  11: 'hyenaspotted',
                  12: 'impala',
                  13: 'koribustard',
                  14: 'lionfemale',
                  15: 'lionmale',
                  16: 'otherbird',
                  17: 'topi',
                  18: 'warthog',
                  19: 'wildebeest',
                  20: 'zebra'}

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")

        #plt.imshow(img.numpy().transpose((1, 2, 0)).squeeze(), cmap="gray")
        imshow(img)
    plt.show()

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    #plt.imshow(img.numpy().transpose((1, 2, 0)).squeeze(), cmap="gray")
    imshow(img)
    plt.show()
    print(f"Label: {label}")
