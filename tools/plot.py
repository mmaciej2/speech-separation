#!/usr/bin/env python3

import os
import numpy as np
from sklearn import cluster
#from sklearn import decomposition
from sklearn import discriminant_analysis
#from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["axes.labelpad"] = 10
matplotlib.rcParams["axes.titlepad"] = 10
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["scatter.marker"] = '.'
matplotlib.rcParams['lines.markersize'] = 2.0

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


def clust_plot(embs, shape, path):
  kmeans = cluster.KMeans(n_clusters=2)
  assigns = kmeans.fit_predict(embs)
  plot_spec(assigns.reshape(shape), path)


def plot_embs(embs, affs, path):
#  pca = decomposition.PCA(n_components=2)
#  reduce_embs = pca.fit_transform(embs.T)

#  tsne = manifold.TSNE(n_components=2)
#  reduce_embs = tsne.fit_transform(embs.T)

  lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)

  full_class_data = np.empty((0, embs.shape[2]))
  full_ids = np.empty(0)
  num_spk = 0
  for i in range(embs.shape[0]):
    emb = embs[i]
    aff = affs[i]
    print("emb shape:", emb.shape, "aff shape:", aff.shape)
    class_data = np.empty((round(np.sum(aff)).astype(int), emb.shape[1]))
    inds = np.nonzero(aff)
    for i in range(len(inds[0])):
      class_data[i,:] = emb[inds[0][i],:]
    print("full_class_data shape:", full_class_data.shape, "class_data shape:", class_data.shape)
    full_class_data = np.append(full_class_data, class_data, axis=0)
    ids = inds[1]+num_spk
    full_ids = np.append(full_ids, ids)
    num_spk += aff.shape[1]
  lda.fit(full_class_data, full_ids)

  embs = embs[0]
  affs = affs[0]
  reduce_embs = lda.transform(embs)

  points = [[[], []] for _ in range(affs.shape[1]+1)]
  for i in range(affs.shape[0]):
    for j in range(affs.shape[1]):
      if affs[i][j] == 1:
        points[j][0].append(reduce_embs[i][0])
        points[j][1].append(reduce_embs[i][1])
        break
    else:
      points[-1][0].append(reduce_embs[i][0])
      points[-1][1].append(reduce_embs[i][1])

  plt.scatter(points[-1][0],points[-1][1], c='k')
  colors = cm.rainbow(np.linspace(0, 1, len(points)-1))
  for i in range(len(points)-1):
    plt.scatter(points[i][0], points[i][1], c=colors[i])

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
