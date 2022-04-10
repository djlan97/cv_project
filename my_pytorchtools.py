import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, loss_based=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0.0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.loss_based = loss_based

    def __call__(self, val_score, model):
        if self.loss_based:
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        # Saves model when better val_score found
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_score:.4f}).  Saving model ...'
                if self.loss_based else f'Validation Accuracy increased ({self.val_acc_max:.4f} --> '
                                        f'{val_score:.4f}).  'f'Saving model ...')
            print()

        torch.save(model.state_dict(), self.path)

        if self.loss_based:
            self.val_loss_min = val_score
        else:
            self.val_acc_max = val_score

