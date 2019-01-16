#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["axes.labelpad"] = 10
matplotlib.rcParams["axes.titlepad"] = 10
matplotlib.rcParams["legend.frameon"] = False

def plot_spec(array, path):
  plt.imshow(np.flipud(array.T))

  plt.tick_params(
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False)
  plt.colorbar(
    aspect=40,
    pad=0.025).ax.tick_params(labelsize='small')

  plt.xlabel('time')
  plt.ylabel('frequency')
  plt.title(os.path.basename(path).split('.')[0].replace('_', ' '))

  plt.savefig(path)

  plt.clf()
  plt.cla()


def plot_loss(*args):
  path = args[-1]

  # Just data
  if np.ndim(args[0]) == 1:
    for i in range(len(args)-1):
      plt.plot(args[i])

  # Data and x values
  elif np.ndim(args[0]) == 2:
    for i in range(len(args)-1):
      x = np.array(args[i])
      if x.shape[0] == 2:
        plt.plot(x[0], x[1])
      else:
        plt.plot(x.T[0], x.T[1])

  # Legend
  if len(args) == 3 and len(args[1][0]) != 0:
    labels = ['train', 'cv']
  else:
    labels = ['train']
  plt.legend(labels)

  plt.title(os.path.basename(path).split('.')[0].replace('_', ' '))
  plt.xlabel('epoch')
  plt.ylabel('avg sample loss')

  plt.tick_params(
    labelsize='x-small',
    direction='in')

  plt.savefig(path)

  plt.clf()
  plt.cla()
