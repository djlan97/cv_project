from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
from plot import imshow, plot_save_results
import time
import os
import copy
from my_pytorchtools import EarlyStopping
from serengeti_dataset import SerengetiDataset
import datetime


def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=10, early_stopping_based_on_loss=True):
    since = time.time()
    print('start: {}'.format(datetime.datetime.now()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.Inf

    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    if early_stopping_based_on_loss:
        print('Early stopping based on loss...')
        early_stopping = EarlyStopping(patience=patience, verbose=True)
    else:
        print('Early stopping based on accuracy...')
        early_stopping = EarlyStopping(patience=patience, verbose=True, loss_based=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracies[phase].append(float(epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if early_stopping_based_on_loss:
                if phase == 'val' and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Better val loss')
            else:
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Better val accuracy')

        print()
        print('Epoch ended: {}'.format(datetime.datetime.now()))

        if early_stopping_based_on_loss:
            early_stopping(epoch_loss, model)
        else:
            early_stopping(epoch_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f} with Acc of {:4f}'.format(best_loss, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses, accuracies


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, _, _) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    early_stopping_based_on_loss = False
    batch_size = 64
    num_workers = 8
    plt.ion()

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

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply(transforms=[transforms.RandomRotation(30)], p=0.3),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=7, sigma=1)], p=0.2),
            transforms.RandomResizedCrop(224, scale=(0.7, 1)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    '''data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}'''

    data_dir = 'data/dataset'
    image_datasets = {x: SerengetiDataset(os.path.join(data_dir, x + '.csv'), os.path.join(data_dir, x),
                                          data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = list(labels_map.values())  # image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes, _, _ = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet34(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 21.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model_ft, losses, accuracies = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, patience=7,
                                               num_epochs=100,
                                               early_stopping_based_on_loss=early_stopping_based_on_loss)

    # losses={'train': [1.7754453440862044, 1.265247638204268, 1.1254793860018253, 1.0177827892026732, 0.934247818535992, 0.8715916809226786, 0.8052162392863206, 0.6538874196953007, 0.5954281648887055, 0.5837510600153889, 0.5553699100390077, 0.5410851184012634, 0.5308041904919913, 0.5155490937642753, 0.4893919150211981, 0.4823248291388154, 0.4781127396571849, 0.4788413030893675, 0.47909358135823693, 0.4744666979515127], 'val': [1.2645114277090346, 1.2016807215554373, 1.101317753110613, 1.0721462156091417, 1.0634040287562778, 1.0571336788790566, 1.0299420314175742, 0.9484106659889221, 0.9432879801307406, 0.950408638375146, 0.9565039847578322, 0.9538189794336046, 0.9600619184119361, 0.9516055864947183, 0.952500884447779, 0.9528498751776558, 0.9607596201556069, 0.9636573370013918, 0.9595207146235875, 0.9605532292808805]}
    # accuracies={'train': [0.47964285714285715, 0.6130952380952381, 0.6547619047619048, 0.6863095238095238, 0.7101785714285714, 0.7344047619047619, 0.7499404761904762, 0.8044642857142857, 0.820595238095238, 0.8257142857142857, 0.8326785714285714, 0.8407142857142857, 0.8427380952380953, 0.8461309523809524, 0.8527380952380952, 0.8586904761904762, 0.856904761904762, 0.8586309523809523, 0.8578571428571429, 0.860297619047619], 'val': [0.6095238095238096, 0.6252380952380953, 0.6676190476190477, 0.6685714285714286, 0.6819047619047619, 0.6795238095238095, 0.6952380952380952, 0.72, 0.7233333333333334, 0.7233333333333334, 0.7276190476190476, 0.7280952380952381, 0.7247619047619047, 0.7304761904761905, 0.7347619047619047, 0.7314285714285714, 0.7295238095238096, 0.73, 0.7333333333333333, 0.73]}

    print(losses)
    print(accuracies)
    print('num_workers: {}'.format(num_workers))
    print('batch_size: {}'.format(batch_size))

    torch.save(model_ft.state_dict(), 'parameters.pt')
    torch.save(model_ft, 'model.pt')

    plot_save_results(losses, accuracies, early_stopping_based_on_loss)

    visualize_model(model_ft)
