# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
from PIL import Image
import pandas as pd
from FeatureExtractor import extractor
import warnings
warnings.filterwarnings('ignore')

# Extract features from the dataset
# data path
path = './data'
feats, image_list = extractor(path)

# test image path
test = './test'
feat_single, image = extractor(test)

scores  = np.dot(feat_single, feats.T)
sort_ind = np.argsort(scores)[0][::-1]
scores = scores[0, sort_ind]

maxres = 10
imlist = [image_list[index] for i, index in enumerate(sort_ind[0:maxres])]
print ("top %d images in order are: " %maxres, imlist)

fig=plt.figure(figsize=(16, 10))
for i in range(len(imlist)):
    sample = imlist[i]
    img = mpimg.imread('./data' + '/' + sample)
    #ax = plt.subplot(figsize)
    ax = fig.add_subplot(2, 5, i+1)
    ax.autoscale()
    plt.tight_layout()
    plt.imshow(img, interpolation='nearest')
    ax.set_title('{:.3f}%'.format(scores[i]))
    ax.axis('off')
plt.show()