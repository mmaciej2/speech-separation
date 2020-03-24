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

def plot_cnn_basis_spec(array, path):
  mag_fft_basis = np.abs(np.fft.fft(array))[:,:int(array.shape[1]/2)-1:-1]
  sort_mat = np.zeros(mag_fft_basis.shape)
  for i in range(mag_fft_basis.shape[1]):
    sort_mat[:,i] = mag_fft_basis[:,i]*i
  sort_inds = np.argsort(np.sum(sort_mat, axis=1)/np.sum(mag_fft_basis, axis=1))
  plot_spec(mag_fft_basis[sort_inds], path+'/basis_vc_mag_spectra.png')
  if array.shape[1] < 512:
    array = np.pad(array, ((0,0),(0,512-array.shape[1])), 'constant')
    mag_fft_basis = np.abs(np.fft.fft(array))[:,:int(array.shape[1]/2)-1:-1]
    plot_spec(mag_fft_basis[sort_inds], path+'/basis_vc_mag_spectra_padded.png')
  return sort_inds

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
