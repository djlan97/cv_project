# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from my_feature_extractor import extractor, save_features, plot_results
import pickle
import pandas as pd

from sklearn.metrics import recall_score


def calculate_recall(class_to_retrieve, retrieved_classes):
    '''
    y_true = np.full(shape=len(retrieved), fill_value=True)

    y_pred = retrieved == class_to_retrieve

    return recall_score(y_true, y_pred)
    '''

    return (retrieved_classes == class_to_retrieve).sum() / 30


def calculate_precision(class_to_retrieve, retrieved_classes):
    '''
    y_true = np.full(shape=len(retrieved), fill_value=True)

    y_pred = retrieved == class_to_retrieve

    return precision_score(y_true, y_pred)
    '''

    return (retrieved_classes == class_to_retrieve).sum() / len(retrieved_classes)


if __name__ == '__main__':
    K = 20
    feats = {}
    image_list = {}
    categories_df = {}
    for x in ['data', 'test']:
        categories_df[x] = pd.read_csv('{}.csv'.format(x)).set_index('name')
        try:
            path = '{}_features.pkl'.format(x)
            with open(path, 'rb') as f:
                feats_dict = pickle.load(f)
            feats[x] = np.array(list(feats_dict.values()))
            image_list[x] = list(feats_dict.keys())
        except FileNotFoundError as _:
            # Extract features from the dataset
            feats[x], image_list[x] = extractor('./{}'.format(x))
            save_features(feats[x], image_list[x], path)

    scores = np.dot(feats['test'], feats['data'].T)
    sort_ind = np.argsort(-scores)

    recall_scores_for_k = np.zeros(K)
    precision_scores_for_k = np.zeros(K)

    for i in range(K):

        maxres = i + 1
        print('K = {}'.format(maxres))
        recall_for_test_im = np.zeros(len(image_list['test']))
        precision_for_test_im = np.zeros(len(image_list['test']))
        for j in range(len(sort_ind)):
            scores[j] = scores[j, sort_ind[j]]
            imlist = [image_list['data'][index] for index in sort_ind[j, 0:maxres]]

            class_to_retrieve = categories_df['test'].iloc[j].category_id
            retrieved_classes = categories_df['data'].loc[imlist].category_id.values

            recall_for_test_im[j] = calculate_recall(class_to_retrieve, retrieved_classes)

            precision_for_test_im[j] = calculate_precision(class_to_retrieve, retrieved_classes)

            '''print("top %d images in order are: " % maxres, imlist)

            plot_results(imlist, scores[j])'''

        recall_scores_for_k[i] = recall_for_test_im.mean()
        precision_scores_for_k[i] = precision_for_test_im.mean()
        print('avg recall is {}'.format(recall_scores_for_k[i]))
        print('avg precision is {}'.format(precision_scores_for_k[i]))

    print('Recall vector for each K: {}'.format(recall_scores_for_k))
    print('Precision vector for each K: {}'.format(precision_scores_for_k))
