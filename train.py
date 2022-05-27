from __future__ import print_function, division

import pandas as pd
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
from early_stopping import EarlyStopping
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

    print('Early stopping based on {}...'.format('loss' if early_stopping_based_on_loss else 'accuracy'))
    early_stopping = EarlyStopping(patience=patience, verbose=True, loss_based=early_stopping_based_on_loss)

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

            # Deep copy the model.
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

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model, losses, accuracies


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    risultati = torch.zeros(21)
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in preds:
                risultati[i] += 1

            output = pd.Series(data=risultati, index=list(labels_map.values()))
            output = output.sort_values(ascending=False)
            print(output)

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

    # Managing parameters for training.
    show_results_example = False
    show_dataloader_example = False
    early_stopping_based_on_loss = False
    early_stopping_patience = 7
    batch_size = 4
    num_workers = 8
    lr = 0.01
    step_size = 20
    model = 'resnet34'
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

    if show_dataloader_example:
        # Get a batch of training data.
        inputs, classes, _, _ = next(iter(dataloaders['train']))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[class_names[x] for x in classes])

    if model == 'resnet18':
        model_ft = models.resnet18(pretrained=True)
    elif model == 'resnet34':
        model_ft = models.resnet34(pretrained=True)
    elif model == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
    elif model == 'googlenet':
        model_ft = models.googlenet(pretrained=True)
    else:
        raise TypeError('Model \'{}\' not avaiable...'.format(model))

    # The size of each output sample is set equal to classes number.
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(image_datasets['train'].img_labels.category_id.unique()))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized.
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of gamma every step_size epochs.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)

    print('Starting {} training on {}...'.format(model, device))

    model_ft, losses, accuracies = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                               patience=early_stopping_patience,
                                               num_epochs=100,
                                               early_stopping_based_on_loss=early_stopping_based_on_loss)

    print(losses)
    print(accuracies)
    print('num_workers: {}'.format(num_workers))
    print('batch_size: {}'.format(batch_size))

    torch.save(model_ft.state_dict(), 'parameters.pt')
    torch.save(model_ft, 'model.pt')

    plot_save_results(losses, accuracies, early_stopping_based_on_loss)

    if show_results_example:
        visualize_model(model_ft)
