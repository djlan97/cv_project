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
from Our_Network import resnetmodmod34
from PIL import ImageFile
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True



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
        transforms.Resize(256),
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
        transforms.Resize(256),
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

dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, pin_memory=True , num_workers=8)
               for x in ['train', 'val', 'train_empty', 'val_empty']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'train_empty', 'val_empty']}

class_names = list(labels_map.values())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=10,
                early_stopping_based_on_loss=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    #best_empty_acc=0.0
    best_loss = np.Inf
    #epoch_acc_empty=0.0
    list_epoch_loss=[]
    list_average_epoch_acc=[]
    list_running_corrects_specie_first=[]
    list_running_corrects_specie_second=[]
    list_running_corrects_specie_final=[]
    list_running_corrects_majorVoting=[]
    list_epoch_acc_descrizione=[]

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
        '''
        if (epoch_acc_empty < 0.55 and epoch > 35)  or epoch < 10:
                torch.save(model.state_dict(), os.path.join('results/models', 'lastbestweightsofempty.pth'))

        if epoch >=10:
            if epoch==10:
                early_stopping.reset()
        '''
        for phase in ['train','val']:
            if phase in ['train']:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            #running_corrects = 0
            running_corrects_specie_first=0
            running_corrects_specie_second = 0
            running_corrects_specie_final = 0
            running_corrects_majorVoting=0
            #running_corrects_empty=0
            running_corrects_descrizione=0


            # Iterate over data.

            for inputs, specie, descrizione, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                specie = specie.to(device)
                descrizione = descrizione.to(device)
                #emptyimg = emptyimg.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    out_specie_first,out_specie_second,out_specie_final,out_descrizione = model(inputs)
                    out_major_voting= (0.2*out_specie_first)+(0.3*out_specie_second)+(0.5*out_specie_final)

                    #_, preds_emptyimg = torch.max(outputs, 1)
                    _, preds_specie_first = torch.max(out_specie_first, 1)
                    _, preds_specie_second = torch.max(out_specie_second, 1)
                    _, preds_specie_final = torch.max(out_specie_final, 1)
                    _, preds_descrizione = torch.max(out_descrizione, 1)
                    _,preds_specie_majorVoting=torch.max(out_major_voting,1)

                    #loss1 = criterion(outputs, emptyimg)
                    loss1 = criterion(out_specie_first, specie)
                    loss2 = criterion(out_specie_second, specie)
                    loss3 = criterion(out_specie_final, specie)
                    loss4 = criterion(out_descrizione, descrizione)

                    loss_tot = loss1 + loss2 + loss3+loss4

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_tot.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_tot.item() * inputs.size(0)
                #if phase == 'val':
                running_corrects_specie_first+=torch.sum(preds_specie_first==specie.data)
                running_corrects_specie_second += torch.sum(preds_specie_second == specie.data)
                running_corrects_specie_final += torch.sum(preds_specie_final == specie.data)
                running_corrects_majorVoting+=torch.sum(preds_specie_majorVoting == specie.data)
                running_corrects_descrizione+=torch.sum(preds_descrizione == torch.max(descrizione.data, 1).indices)

                '''
                running_corrects += torch.sum(torch.logical_and(
                    torch.logical_and(preds_emptyimg == emptyimg.data, preds_specie == specie.data),
                    preds_descrizione == torch.max(descrizione.data, 1).indices))
                '''

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_specie = running_corrects_specie_final.double() / dataset_sizes[phase]
            epoch_acc_specie_MajorVoting=running_corrects_majorVoting.double() / dataset_sizes[phase]
            epoch_acc_descrizione = running_corrects_descrizione.double() / dataset_sizes[phase]
            average_epoch_acc = (epoch_acc_specie_MajorVoting + epoch_acc_descrizione) / 2

            if phase == 'train':
                list_epoch_loss.append(epoch_loss)

            if phase == 'val':
                #epoch_acc_specie=running_corrects_specie_final.double() / dataset_sizes[phase]
                epoch_acc_specie_first = running_corrects_specie_first.double() / dataset_sizes[phase]
                epoch_acc_specie_second = running_corrects_specie_second.double() / dataset_sizes[phase]
                #epoch_acc_specie_MajorVoting = running_corrects_majorVoting.double() / dataset_sizes[phase]
                list_running_corrects_specie_first.append(epoch_acc_specie_first)
                list_running_corrects_specie_second.append(epoch_acc_specie_second)
                list_running_corrects_specie_final.append(epoch_acc_specie)
                list_running_corrects_majorVoting.append(epoch_acc_specie_MajorVoting)


                #epoch_acc_descrizione = running_corrects_descrizione.double() / dataset_sizes[phase]
                #average_epoch_acc=(epoch_acc_specie+epoch_acc_descrizione)/2
                list_epoch_acc_descrizione.append(epoch_acc_descrizione)
                list_average_epoch_acc.append(average_epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f} Acc_Specie_First: {:.4f} Acc_Specie_Second: {:.4f} Acc_Specie_Final: {:.4f} Acc_Specie_MajorVoting: {:.4f} Acc_Descrizione: {:.4f}'.format(phase, epoch_loss, average_epoch_acc,epoch_acc_specie_first,epoch_acc_specie_second,epoch_acc_specie,epoch_acc_specie_MajorVoting,epoch_acc_descrizione))
            else:

                print('{} Loss: {:.4f} Acc: {:.4f} Acc_MajorVoting: {:.4f}'.format(phase, epoch_loss, average_epoch_acc,epoch_acc_specie_MajorVoting))

            # deep copy the model
            if early_stopping_based_on_loss:
                if phase == 'val' and epoch_loss < best_loss:
                    best_acc = average_epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(),os.path.join('results/models','lastbestweightsLightBranchesResNet34Backbone.pth'))
                    print('Better val loss')
            else:
                if phase == 'val' and average_epoch_acc > best_acc:
                    best_acc = average_epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join('results/models', 'lastbestweightsLightBranchesResNet34Backbone.pth'))

                    print('Better val accuracy')

        print()



        if early_stopping_based_on_loss:
            early_stopping(epoch_loss, model)
        else:
            early_stopping(average_epoch_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print(list_epoch_loss)
    print(list_average_epoch_acc)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f} with Acc of {:4f}'.format(best_loss, best_acc))

    # load best model weights
    '''
    f = open('lossResnet34withBracnhes.txt', 'w')
    for ele in list_epoch_loss:
        f.write(ele + '\n')

    f.close()
    f = open('averageEpochAccResnet34withBracnhes.txt', 'w')
    for ele in list_average_epoch_acc:
        f.write(ele + '\n')

    f.close()
    f = open('MajorVotingAccResnet34withBracnhes.txt', 'w')
    for ele in list_running_corrects_majorVoting:
        f.write(ele + '\n')

    f.close()
    f = open('FirstOutputAccResnet34withBracnhes.txt', 'w')
    for ele in list_running_corrects_specie_first:
        f.write(ele + '\n')

    f.close()
    f = open('SecondOutputAccResnet34withBracnhes.txt', 'w')
    for ele in list_running_corrects_specie_second:
        f.write(ele + '\n')

    f.close()
    f = open('FinalOutputAccResnet34withBracnhes.txt', 'w')
    for ele in list_running_corrects_specie_final:
        f.write(ele + '\n')

    f.close()
    f = open('DescriptionAccResnet34withBracnhes.txt', 'w')
    for ele in list_epoch_acc_descrizione:
        f.write(ele + '\n')

    f.close()
    '''
    model.load_state_dict(best_model_wts)
    return model


def initializeWeights34(modelmod34):
  modelmod34.conv1.load_state_dict(torch.load('resnet34conv1.pth'))
  modelmod34.layer1.load_state_dict(torch.load('resnet34layer1.pth'))
  modelmod34.layer2.load_state_dict(torch.load('resnet34layer2.pth'))
  modelmod34.layer42.load_state_dict(torch.load('resnet34layer4.pth'))
  modelmod34.layer31.load_state_dict(torch.load('resnet50layer2.pth'))
  modelmod34.layer41.load_state_dict(torch.load('resnet50layer3.pth'))
  modelmod34.layer51.load_state_dict(torch.load('resnet50layer4.pth'))


model_ft = resnetmodmod34()

model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)


if __name__ == "__main__":
    resntet50pre = models.resnet50(pretrained=True)
    resnetpre = models.resnet34(pretrained=True)
    torch.save(resnetpre.conv1.state_dict(), 'resnet34conv1.pth')
    torch.save(resnetpre.layer1.state_dict(), 'resnet34layer1.pth')
    torch.save(resnetpre.layer2.state_dict(), 'resnet34layer2.pth')
    torch.save(resnetpre.layer3.state_dict(), 'resnet34layer3.pth')
    torch.save(resnetpre.layer4.state_dict(), 'resnet34layer4.pth')
    torch.save(resnetpre.fc.state_dict(), 'resnet34layerfc.pth')
    torch.save(resntet50pre.conv1.state_dict(), 'resnet50conv1.pth')
    torch.save(resntet50pre.layer1.state_dict(), 'resnet50layer1.pth')
    torch.save(resntet50pre.layer2.state_dict(), 'resnet50layer2.pth')
    torch.save(resntet50pre.layer3.state_dict(), 'resnet50layer3.pth')
    torch.save(resntet50pre.layer4.state_dict(), 'resnet50layer4.pth')
    initializeWeights34(model_ft)

    #model_ft.load_state_dict(copy.deepcopy(torch.load('results/models/lastbestweightsofempty.pth')) )
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           patience=7,num_epochs=60, early_stopping_based_on_loss=False)


    torch.save(model_ft, '34BackboneWithBranches.pth')

