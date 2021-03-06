import torch
import torch.nn as nn
import pickle
import numpy as np
from torchvision import models
import time
import os
from PIL import Image
from skimage.transform import resize
from numpy import linalg
from torch.autograd import Variable


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def create_retirval_net(model, finetuned):
    if model == 'resnet18':
        ret_model = models.resnet18(pretrained=True)
    elif model == 'resnet34':
        ret_model = models.resnet34(pretrained=True)
    elif model == 'resnet50':
        ret_model = models.resnet50(pretrained=True)
    elif model == 'googlenet':
        ret_model = models.googlenet(pretrained=True)
    else:
        raise Exception('Model \'{}\' not avaiable...'.format(model))

    num_ftrs = ret_model.fc.in_features

    if finetuned:
        # Here the size of each output sample is set to 21.
        ret_model.fc = nn.Linear(num_ftrs, 21)

        ret_model.load_state_dict(torch.load('../results/{}/parameters.pt'.format(model)))

    # Remove last FC layers.
    ret_model = torch.nn.Sequential(*(list(ret_model.children())[:-1]))

    return ret_model, num_ftrs


def extractor(ret_model, num_ftrs, data):
    since = time.time()

    # Read images images from a directory.
    list_imgs_names = os.listdir(data)

    # Create an array to store features.
    images_number = len(list_imgs_names)
    fea_all = np.zeros((images_number, num_ftrs))

    # Define empy array to store image names.
    image_all = []

    # Extract features.
    for ind, img_name in enumerate(list_imgs_names):
        print('{}/{} -> {}'.format(ind + 1, len(list_imgs_names), img_name))

        img_path = os.path.join(data, img_name)
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224))
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_np = Variable(image_np.unsqueeze(0))  # bs, c, h, w
        image_np = image_np.to(device)

        # Pass image to the model.
        fea = ret_model(image_np)
        fea = fea.squeeze()
        fea = fea.cpu().data.numpy()
        fea = fea.reshape((1, num_ftrs))
        fea = fea / linalg.norm(fea)
        fea_all[ind] = fea
        image_all.append(img_name)

    time_elapsed = time.time() - since

    print('Feature extraction complete in {:.02f}s'.format(time_elapsed % 60))

    return fea_all, image_all


def save_features(feats, image_list, path):
    # Save feature as a dictionary having image name as kay and feature vector as value.
    feats_dict = {image_list[i]: feats[i] for i in range(len(image_list))}

    with open(path, 'wb') as f:
        pickle.dump(feats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    finetuned = False
    model = 'newBackbone50SecondStrategy'

    if model in ['Backbone34FirstStrategy', 'Backbone34SecondStrategy',
                 'NewBackbone50FirstStrategy', 'NewBackbone50SecondStrategy']:
        ret_model = torch.load('models/{}.pth'.format(model))
        if model in ['Backbone34FirstStrategy', 'Backbone34SecondStrategy']:
            num_ftrs = 96
        else:
            num_ftrs = 256

    else:
        ret_model, num_ftrs = create_retirval_net(model, finetuned)

    # Set the model train False since we are using our feature extraction network.
    ret_model.train(False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ret_model.to(device)

    feats = {}
    image_list = {}

    # For each image in test and data folder extratc feature vector and save it in a specific folder.
    for x in ['test', 'data']:
        path = 'features/{}{}/{}_features.pkl'.format(model, '_finetuned' if finetuned else '', x)
        feats[x], image_list[x] = extractor(ret_model, num_ftrs, './data/{}'.format(x))
        save_features(feats[x], image_list[x], path)
