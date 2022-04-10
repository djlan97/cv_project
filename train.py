from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from my_pytorchtools import EarlyStopping
from serengeti_dataset import SerengetiDataset
from Our_Network import resnetmod34
from PIL import ImageFile
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


def imshow(inp, title=None):
    # Imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, optimizer_empty, scheduler, scheduler_empty, num_epochs=100, patience=10,
                early_stopping_based_on_loss=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.Inf

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
        # TODO Rimuovi train e rinomina variabili
        for phase in ['train', 'val', 'train_empty', 'val_empty', 'train', 'val']:
            if phase in ['train', 'train_empty']:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase in ['train_empty', 'val_empty']:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer_empty.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train_empty'):
                        outputs, _, _ = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train_empty':
                            loss.backward()
                            optimizer_empty.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train_empty':
                    scheduler_empty.step()
            else:  # if phase in ['train', 'val']
                for inputs, specie, descrizione, emptyimg in dataloaders[phase]:
                    inputs = inputs.to(device)
                    specie = specie.to(device)
                    descrizione = descrizione.to(device)
                    emptyimg = emptyimg.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, out_specie, out_descrizione = model(inputs)

                        _, preds_emptyimg = torch.max(outputs, 1)
                        _, preds_specie = torch.max(out_specie, 1)
                        _, preds_descrizione = torch.max(out_descrizione, 1)

                        loss1 = criterion(outputs, emptyimg)
                        loss2 = criterion(out_specie, specie)
                        loss3 = criterion(out_descrizione, descrizione)

                        loss_tot = loss1 + loss2 + loss3

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_tot.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss_tot.item() * inputs.size(0)

                    running_corrects += torch.sum(torch.logical_and(
                        torch.logical_and(preds_emptyimg == emptyimg.data, preds_specie == specie.data),
                        preds_descrizione == torch.max(descrizione.data, 1).indices))

                if phase == 'train':
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, _, specie, descrizione) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            specie = specie.to(device)
            descrizione = descrizione.to(device)

            outputs, out_specie, out_descrizione = model(inputs)

            _, preds_emptyimg = torch.max(outputs, 1)
            preds_specie_perc, preds_specie = torch.topk(out_specie, 3, 1)
            _, preds_descrizione = torch.max(out_descrizione, 1)

            preds_specie_perc = preds_specie_perc/torch.sum(preds_specie_perc,dim=1,keepdim=True)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(
                    'predicted: {} {}, {} {}, {} {}'.format(class_names[preds_specie[j][0]],preds_specie_perc[j, 0], class_names[preds_specie[j][1]],preds_specie_perc[j, 1],
                                                   class_names[preds_specie[j][2]],preds_specie_perc[j, 2]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    torch.cuda.empty_cache()
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
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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
        'train_empty': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_empty': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_dir = 'data/dataset'
    image_datasets = {x: SerengetiDataset(os.path.join(data_dir, x + '.csv'), os.path.join(data_dir, x),
                                          data_transforms[x])
                      for x in ['train', 'val']}

    data_dir_empty = 'data/empty_dataset'

    image_datasets.update({x: datasets.ImageFolder(os.path.join(data_dir_empty, y),
                                                   data_transforms[x])
                           for y, x in zip(['train', 'val'], ['train_empty', 'val_empty'])})

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=8)
                   for x in ['train', 'val', 'train_empty', 'val_empty']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'train_empty', 'val_empty']}

    class_names = list(labels_map.values())  # image_datasets['train'].classes

    '''
    # Get a batch of training data
    inputs, specie, descrizione, emptyimg = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in specie])

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 21.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))'''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_ft = torch.load('our_modelmod34.pth')

    model_ft = resnetmod34()

    model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft_empty = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler_empty = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, optimizer_ft_empty, exp_lr_scheduler,
                           exp_lr_scheduler_empty, patience=5,
                           num_epochs=100, early_stopping_based_on_loss=False)

    torch.save(model_ft, os.path.join('models', 'our_model.pth'))

    visualize_model(model_ft)
