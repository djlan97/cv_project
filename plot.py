import numpy as np
from matplotlib import pyplot as plt


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


def plot_save_results(losses, accuracies, early_stopping_based_on_loss=True):
    # visualize the loss as the network trained
    fig_loss = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(losses['train']) + 1), losses['train'], label='Training Loss')
    plt.plot(range(1, len(losses['val']) + 1), losses['val'], label='Validation Loss')

    # find position of lowest validation loss
    if early_stopping_based_on_loss:
        minposs = losses['val'].index(min(losses['val'])) + 1
    else:
        minposs = accuracies['val'].index(max(accuracies['val'])) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.5)  # consistent scale
    plt.xticks(range(0, len(losses['train']) + 1))
    plt.xlim(0, len(losses['train']) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_loss.savefig('loss_plot.png', bbox_inches='tight')

    # plot accuracy
    fig_acc = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(accuracies['train']) + 1), accuracies['train'], label='Training Accuracy')
    plt.plot(range(1, len(accuracies['val']) + 1), accuracies['val'], label='Validation Accuracy')

    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    #plt.ylim(0, 0.5)  # consistent scale
    plt.xticks(range(0, len(losses['train']) + 1))
    plt.xlim(0, len(accuracies['train']) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_acc.savefig('acc_plot.png', bbox_inches='tight')
