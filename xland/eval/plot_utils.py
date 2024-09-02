from datetime import datetime

from matplotlib import pyplot as plt
import os

import numpy as np
FIG_DPI = 200


def plot_mean_std(mean, std, ax, label, **kwargs):
    X = np.arange(len(mean))
    ax.plot(X, mean, label=label, **kwargs)
    ax.fill_between(X, mean - std, mean + std, alpha=0.2)

def plot_mean_std_xy(X, mean, std, ax, label, **kwargs):
    if 'color' in kwargs:
       kwargs2 = {'color': kwargs['color']}
    else:
       kwargs2 = {}
    ax.plot(X, mean, label=label, **kwargs)
    ax.fill_between(X, mean - std, mean + std, alpha=0.2, **kwargs2)

def scatter_plot(mean, ax, label):
    X = np.arange(len(mean))
    ax.scatter(X, mean, label=label)


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize='large'):
  """Helper function for decorating plots."""
  # Hide the right and top spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  # Deal with ticks and the blank space at the origin
  ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
  ax.spines['left'].set_position(('outward', hrect))
  ax.spines['bottom'].set_position(('outward', wrect))
  return ax



def _annotate_and_decorate_axis(ax,
                                labelsize='x-large',
                                ticklabelsize='x-large',
                                xticks=None,
                                xticklabels=None,
                                yticks=None,
                                legend=False,
                                grid_alpha=0.2,
                                legendsize='x-large',
                                xlabel='',
                                ylabel='',
                                wrect=10,
                                hrect=10):
  """Annotates and decorates the plot."""
  ax.set_xlabel(xlabel, fontsize=labelsize)
  ax.set_ylabel(ylabel, fontsize=labelsize)
  if xticks is not None:
    ax.set_xticks(ticks=xticks)
    ax.set_xticklabels(xticklabels)
  if yticks is not None:
    ax.set_yticks(yticks)
  ax.grid(True, alpha=grid_alpha)
  ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
  if legend:
    ax.legend(fontsize=legendsize)
  return ax