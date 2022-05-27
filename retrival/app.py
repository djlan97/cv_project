import numpy as np
import pandas as pd
from retrieval import load_features
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances, euclidean_distances
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':

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

    k=10
    model='resnet34_finetuned'
    feats = {}
    categories_df = {}
    images_dict={}
    for x in ['data', 'test']:
        categories_df[x] = pd.read_csv('data/{}.csv'.format(x)).set_index('name')
        path = 'features/{}/{}_features.pkl'.format(model, x)
        feats[x], images_dict[x]  = load_features(path)

    # Insert a test image in query folder to try the retrieval system.
    list_imgs_names = os.listdir('query')

    scores = cosine_distances(feats['test'][images_dict['test'].index(list_imgs_names[0])].reshape(1,-1), feats['data'])  # np.dot(feats['test'], feats['data'].T)

    # Get indices of the images with the best scores.
    sort_ind = np.argsort(scores).reshape(-1)

    scores = scores[0, sort_ind].reshape(-1)

    imlist = [images_dict['data'][index] for i, index in enumerate(sort_ind[0:k])]
    cat_list=[categories_df['data'].iloc[index].category_id for i, index in enumerate(sort_ind[0:k])]

    print("top %d images in order are: " % k, imlist)

    # Plot results.
    fig = plt.figure(figsize=(16, 10))
    for i in range(len(imlist)):
        sample = imlist[i]
        img = mpimg.imread('./data/data' + '/' + sample)
        # ax = plt.subplot(figsize)
        ax = fig.add_subplot(2, 5, i + 1)
        ax.autoscale()
        plt.tight_layout()
        plt.imshow(img, interpolation='nearest')
        ax.set_title('{:.2f}% - {}'.format((1-scores[i])*100,labels_map[cat_list[i]]))
        print(scores[i])
        ax.axis('off')
    plt.show()
    fig.savefig('example')