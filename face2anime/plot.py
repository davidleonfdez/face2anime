from face2anime.losses import CritPredsTracker, MultiCritPredsTracker
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


__all__ = ['plot_c_preds', 'plot_multi_c_preds']


def plot_c_preds(c_preds_tracker:CritPredsTracker):
    preds_xs = range(len(c_preds_tracker.real_preds))
    sns.lineplot(x=preds_xs, y=c_preds_tracker.fake_preds.cpu(), label='Fake preds')
    ax=sns.lineplot(x=preds_xs, y=c_preds_tracker.real_preds.cpu(), label='Real preds')
    ax.set_xlabel('Number of batches')
    ax.set_ylabel('Critic preds')
    return ax


def plot_multi_c_preds(multi_c_preds_tracker:MultiCritPredsTracker, titles:List[str]):
    axs = []
    for c_preds_tracker, title in zip(multi_c_preds_tracker.trackers, titles):
        ax=plot_c_preds(c_preds_tracker)
        axs.append(ax)
        plt.legend(title=title)
        plt.figure()
    return axs
