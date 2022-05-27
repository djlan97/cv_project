import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import statistics


def calculate_recall(class_to_retrieve, retrieved_classes):
    return (retrieved_classes == class_to_retrieve).sum() / 30


def calculate_precision(class_to_retrieve, retrieved_classes):
    return (retrieved_classes == class_to_retrieve).sum() / len(retrieved_classes)


def load_features(path):
    try:
        with open(path, 'rb') as f:
            feats_dict = pickle.load(f)
            return np.array(list(feats_dict.values())), list(feats_dict.keys())
    except FileNotFoundError as _:
        raise Exception('Selected model is not avaiable or file \'{}\' doesn\'t exist'.format(path))


def plot_save_scores(recall_scores, precision_scores, k):
    scores = {'recall': recall_scores, 'precision': precision_scores}

    for score in scores.keys():
        fig = plt.figure(figsize=(10, 8))
        for key in scores[score].keys():
            plt.plot(range(1, k + 1), scores[score][key], label=key)

        plt.xlabel('K')
        plt.ylabel(score)
        plt.xticks(range(0, k + 1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()
        fig.savefig('{}_plot.png'.format(score), bbox_inches='tight')


if __name__ == '__main__':
    # The maximum number of results to retrieve.
    k = 20

    # List of models to compare.
    models = ['resnet50', 'resnet50_finetuned', 'resnet18', 'resnet18_finetuned', 'resnet34', 'resnet34_finetuned',
              'googlenet', 'googlenet_finetuned', 'Backbone34FirstStrategy', 'Backbone34SecondStrategy',
              'NewBackbone50FirstStrategy', 'NewBackbone50SecondStrategy']

    # Dictionaries containing recall and precision scores for each model and for each k.
    recall_scores_for_k = {}
    precision_scores_for_k = {}

    recall_scores_for_classes_and_models = {}
    precision_scores_for_classes_and_models = {}

    for model in models:
        print('Retrival results for {} model'.format(model))
        feats = {}
        categories_df = {}

        # Loading features contained in a file for data and query images.
        for x in ['data', 'test']:
            categories_df[x] = pd.read_csv('data/{}.csv'.format(x)).set_index('name')
            path = 'features/{}/{}_features.pkl'.format(model, x)
            feats[x], _ = load_features(path)

        scores = euclidean_distances(feats['test'], feats['data'])  # np.dot(feats['test'], feats['data'].T)

        # Get indices of the images with the best scores.
        sort_ind = np.argsort(scores)

        recall_scores_for_k[model] = np.zeros(k)
        precision_scores_for_k[model] = np.zeros(k)

        recall_scores_for_classes_and_models[model] = {i: [] for i in range(21)}
        precision_scores_for_classes_and_models[model] = {i: [] for i in range(21)}

        # For each k scores are calculated.
        for i in range(k):

            maxres = i + 1
            print('K = {}'.format(maxres))
            recall_for_test_im = np.zeros(len(sort_ind))
            precision_for_test_im = np.zeros(len(sort_ind))

            recall_scores_for_classes = {i: [] for i in range(21)}
            precision_scores_for_classes = {i: [] for i in range(21)}

            for j, index in enumerate(sort_ind):
                class_to_retrieve = categories_df['test'].iloc[j].category_id
                retrieved_classes = categories_df['data'].iloc[index[0:maxres]].category_id.values

                recall_for_test_im[j] = calculate_recall(class_to_retrieve, retrieved_classes)

                precision_for_test_im[j] = calculate_precision(class_to_retrieve, retrieved_classes)

                recall_scores_for_classes[class_to_retrieve].append(recall_for_test_im[j])
                precision_scores_for_classes[class_to_retrieve].append(precision_for_test_im[j])

            # Take the average of all query images scores for every k.
            recall_scores_for_k[model][i] = recall_for_test_im.mean()
            precision_scores_for_k[model][i] = precision_for_test_im.mean()

            for key in recall_scores_for_classes.keys():
                recall_scores_for_classes_and_models[model][key].append(statistics.mean(recall_scores_for_classes[key]))
                precision_scores_for_classes_and_models[model][key].append(
                    statistics.mean(precision_scores_for_classes[key]))

            print('avg recall is {}'.format(recall_scores_for_k[model][i]))
            print('avg precision is {}'.format(precision_scores_for_k[model][i]))
            print()

        print('Recall vector for each K: {}'.format(recall_scores_for_k[model]))
        print('Precision vector for each K: {}'.format(precision_scores_for_k[model]))

    plot_save_scores(recall_scores_for_k, precision_scores_for_k, k)
