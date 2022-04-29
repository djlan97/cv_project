# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from numpy import linalg as LA
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import time
from torchvision import transforms
import copy
import os
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings

warnings.filterwarnings('ignore')


# Create the network to extract the features
class MyResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, transform_input=False):
        super(MyResNetFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.fc = resnet.fc
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.conv1(x)
        # 149 x 149 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # 147 x 147 x 32
        x = self.layer1(x)
        # 147 x 147 x 64
        x = self.layer2(x)
        # 73 x 73 x 64
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=7)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Import pre-trained model from by using torchvision package
model = torchvision.models.resnet50(pretrained=True)  # resnet 50 model is imported

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 21.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 21)
model.load_state_dict(torch.load('../results/resnet50/resnet50_parameters.pt'))



# set the model train False since we are using our feature extraction network
model.train(False)

# Set our model with pre-trained model
my_resnet = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)#MyResNetFeatureExtractor(model).to(device)


def extractor(data):
    since = time.time()
    # read images images from a directory
    # root = './index/'
    list_imgs_names = os.listdir(data)
    # list_imgs_names
    # create an array to store features
    N = len(list_imgs_names)
    fea_all = np.zeros((N, 2048))
    # define empy array to store image names
    image_all = []
    # extract features
    for ind, img_name in enumerate(list_imgs_names):
        # print(img_name)
        img_path = os.path.join(data, img_name)
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224))
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_np = Variable(image_np.unsqueeze(0))  # bs, c, h, w
        image_np = image_np.to(device)
        fea = my_resnet(image_np)
        fea = fea.squeeze()
        fea = fea.cpu().data.numpy()
        fea = fea.reshape((1, 2048))
        fea = fea / LA.norm(fea)
        fea_all[ind] = fea
        image_all.append(img_name)

    time_elapsed = time.time() - since

    print('Feature extraction complete in {:.02f}s'.format(time_elapsed % 60))

    return fea_all, image_all


def save_features(feats, image_list,path):
    feats_dict = {image_list[i]: feats[i] for i in range(len(image_list))}

    with open(path, 'wb') as f:
        pickle.dump(feats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def plot_results(imlist,scores):
    fig = plt.figure(figsize=(16, 10))
    for i in range(len(imlist)):
        sample = imlist[i]
        img = mpimg.imread('./data' + '/' + sample)
        # ax = plt.subplot(figsize)
        ax = fig.add_subplot(2, 5, i + 1)
        ax.autoscale()
        plt.tight_layout()
        plt.imshow(img, interpolation='nearest')
        ax.set_title('{:.3f}%'.format(scores[i]))
        ax.axis('off')
    plt.show()

def plot_save_results(recalls, precisions,k):
    # visualize the loss as the network trained
    fig_loss = plt.figure(figsize=(10, 8))
    plt.plot(range(1, k+1), recalls, label='Recall')
    plt.plot(range(1, k+1), precisions, label='Precision')

    plt.xlabel('k')
    plt.xticks(range(0, k+1))

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 1)

    '''
    
    #plt.ylim(0, 0.5)  # consistent scale

    plt.xlim(0, k)  # consistent scale
    '''
    plt.show()
    fig_loss.savefig('plot.png', bbox_inches='tight')

