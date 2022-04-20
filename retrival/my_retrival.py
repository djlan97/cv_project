# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from my_feature_extractor import extractor, save_features, plot_results
import pickle

if __name__ == '__main__':

    try:
        with open('features.pkl', 'rb') as fp:
            feats_dict = pickle.load(fp)
            feats = np.array(list(feats_dict.values()))
            image_list = list(feats_dict.keys())
    except FileNotFoundError as _:
        # Extract features from the dataset
        feats, image_list = extractor('./data')
        save_features(feats, image_list)

    # test image path
    test = './test'
    feat_single, image = extractor(test)

    scores = np.dot(feat_single, feats.T)
    sort_ind = np.argsort(scores)[0][::-1]
    scores = scores[0, sort_ind]

    maxres = 10
    imlist = [image_list[index] for i, index in enumerate(sort_ind[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)

    plot_results(imlist,scores)

    '''
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
    plt.show()'''
