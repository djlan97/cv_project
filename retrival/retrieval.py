import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

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


def plot_save_scores(recall_scores,precision_scores,k):

    scores = {'recall' : recall_scores, 'precision' : precision_scores}

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

    k = 20
    models = ['resnet50','resnet50_finetuned']
    recall_scores_for_k={}
    precision_scores_for_k={}
    for model in models:

        feats = {}
        # image_list = {}
        categories_df = {}

        for x in ['data', 'test']:
            categories_df[x] = pd.read_csv('data/{}.csv'.format(x)).set_index('name')
            path = 'features/{}/{}_features.pkl'.format(model, x)
            feats[x], _ = load_features(path)

        scores = np.dot(feats['test'], feats['data'].T)
        sort_ind = np.argsort(-scores)

        recall_scores_for_k[model] = np.zeros(k)
        precision_scores_for_k[model] = np.zeros(k)

        for i in range(k):

            maxres = i + 1
            print('K = {}'.format(maxres))
            recall_for_test_im = np.zeros(len(scores))
            precision_for_test_im = np.zeros(len(scores))
            for j in range(len(sort_ind)):
                scores[j] = scores[j, sort_ind[j]]

                class_to_retrieve = categories_df['test'].iloc[j].category_id
                retrieved_classes = categories_df['data'].iloc[sort_ind[j, 0:maxres]].category_id.values

                recall_for_test_im[j] = calculate_recall(class_to_retrieve, retrieved_classes)

                precision_for_test_im[j] = calculate_precision(class_to_retrieve, retrieved_classes)


            recall_scores_for_k[model][i] = recall_for_test_im.mean()
            precision_scores_for_k[model][i] = precision_for_test_im.mean()
            print('avg recall is {}'.format(recall_scores_for_k[model][i]))
            print('avg precision is {}'.format(precision_scores_for_k[model][i]))
            print()

        print('Recall vector for each K: {}'.format(recall_scores_for_k[model]))
        print('Precision vector for each K: {}'.format(precision_scores_for_k[model]))

    plot_save_scores(recall_scores_for_k,precision_scores_for_k, k)



