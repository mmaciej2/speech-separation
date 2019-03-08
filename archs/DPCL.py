#/usr/bin/env python3

import torch
import sys
import os
import numpy as np
import collections
import re
from sklearn import cluster

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate, numpy_type_map
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('tools')
import plot


# Define collating function (that constructs packed sequences from a batch)
class Collator():

  def __init__(self, sort_key):
    self.key = sort_key
    if not self.key:
      print("Warning: you have not provided a sort key.")
      print("  If you are using RNNs with variable-length input, you must")
      print("  provide the key for element in each sample that is the input")
      print("  of variable length.")

  def __call__(self, batch):
    if not self.key:
      return default_collate(batch)
    else:
      error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
      elem_type = type(batch[0])
      if isinstance(batch[0], collections.Mapping):
        sort_inds = np.argsort(np.array([len(d[self.key]) for d in batch]))[::-1]
        return {key: self.__call__([batch[i][key] for i in sort_inds]) for key in batch[0]}
      elif elem_type.__module__ == 'numpy' and elem_type.__name__ == 'ndarray':
        # array of string classes and object
        if re.search('[SaUO]', batch[0].dtype.str) is not None:
          raise TypeError(error_msg.format(elem.dtype))
        return pack_sequence([(torch.from_numpy(b)).float() for b in batch])
      else:
        return default_collate(batch)

# Define dataset
class TrainSet(Dataset):

  def __init__(self, datadir, location=""):
    filelist = datadir+"/feats_dpcl_train.scp"
    if location:
      with open(filelist) as F:
        indir = os.path.dirname(F.readline().rstrip().split(' ')[1])+'/'
      print("rsync -r --bwlimit=10000 "+indir+" "+location)
      os.system("rsync -r --bwlimit=10000 "+indir+" "+location)
      self.list = [location+'/'+line.split(' ')[0]+'.npz' for line in open(filelist)]
    else:
      self.list = [line.rstrip('\n').split(' ')[1] for line in open(filelist)]

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    feat = np.load(self.list[idx])
    mix_mag_spec = feat['mix'].astype('float32')
    affinity = feat['affinity'].astype('float32')
    return {'mix': mix_mag_spec, 'affinity': affinity}

  def collator(self, batch):
    return {key: default_collate([subelt for d in batch for subelt in d[key]]) for key in batch[0]}

class TestSet(Dataset):

  def __init__(self, datadir):
    self.list = [line.rstrip('\n').split(' ')[1] for line in open(datadir+"/feats_test.scp")]
    self.num_spks = [int(line.rstrip('\n').split(' ')[1]) for line in open(datadir+"/utt2num_spk")]
#    self.collator = Collator('mix')
    self.collator = default_collate

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.abs(np.load(self.list[idx])['mix']).transpose()

    sample = {'mix': mix_mag_spec, 'name': os.path.basename(self.list[idx]), 'num_spk': self.num_spks[idx]}
    return sample

# define nnet
class SepDNN(nn.Module):
  def __init__(self, gpuid, **kwargs):
    super(SepDNN, self).__init__()

    self.gpuid = gpuid

    if 'feat_dim' in kwargs.keys():
      self.feat_dim = int(kwargs['feat_dim'])
    else:
      self.feat_dim = 129

    if 'emb_dim' in kwargs.keys():
      self.emb_dim = int(kwargs['emb_dim'])
    else:
      self.emb_dim = 40

    for key in kwargs.keys():
      print('modelparam:', key, kwargs[key])

    self.blstm = nn.LSTM(self.feat_dim, 600, num_layers=2, bidirectional=True, batch_first=True)

    self.bn = nn.BatchNorm1d(600*2)

    self.lin = nn.Linear(600*2, self.feat_dim*self.emb_dim)

  def init_hidden(self, batch_size):
    if self.gpuid > -1:
      return (torch.randn(2*2, batch_size, 600).cuda(),
              torch.randn(2*2, batch_size, 600).cuda())
    else:
      return (torch.randn(2*2, batch_size, 600),
              torch.randn(2*2, batch_size, 600))

  def forward(self, x):
    # x: tensor of shape (batch, seq_length, feat_dim)

    x, self.hidden = self.blstm(x, self.hidden)
    # x: tensor of shape (batch, seq_length, 600*2)

    x = self.bn(x.permute(0,2,1).contiguous()).permute(0,2,1)
    # x: tensor of shape (batch, seq_length, 600*2)

    x = self.lin(x)
    # x: tensor of shape (batch, seq_length, feat_dim*emb_dim)

#    x = F.sigmoid(x)
    x = F.tanh(x)
    # x: tensor of shape (batch, seq_length, feat_dim*emb_dim)

#    x = x.permute(0,2,1).contiguous()
#    # x: tensor of shape (batch, feat_dim*emb_dim, seq_length)
#
#    x = x.view((x.shape[0], self.emb_dim, self.feat_dim, x.shape[2]))
#    # x: tensor of shape (batch, emb_dim, feat_dim, seq_length)

    x = x.view((x.shape[0], x.shape[1], self.feat_dim, self.emb_dim))
    # x: tensor of shape (batch, seq_length, feat_dim, emb_dim)

#    x = x.view((x.shape[0], self.feat_dim, x.shape[1], self.emb_dim))
#    # x: tensor of shape (batch, feat_dim, seq_length, emb_dim)

    return x.contiguous()

def compute_cv_loss(model, epoch, batch_sample, plotdir=""):
  if plotdir:
    loss, norm = compute_loss(model, epoch, batch_sample, plotdir)
  else:
    loss, norm = compute_loss(model, epoch, batch_sample)
  return loss, norm

# define training pass
def compute_loss(model, epoch, batch_sample, plotdir=""):
  model.zero_grad()

  mix = batch_sample['mix'].cuda() # shape: (batch, seq_length, feat_dim)
#  affinity = batch_sample['affinity'].permute(0,2,1,3).cuda() # shape: (batch, feat_dim, seq_length, num_spk)
  affinity = batch_sample['affinity'].cuda() # shape: (batch, seq_length, feat_dim, num_spk)
#  print("aff[0,0]:", affinity[0,0,0,:].detach().cpu().numpy())
#  print("aff[0,1]:", affinity[0,0,1,:].detach().cpu().numpy())
#  print("aff[1,0]:", affinity[0,1,0,:].detach().cpu().numpy())
#  print("aff[1,1]:", affinity[0,1,1,:].detach().cpu().numpy())

  loss = 0
  norm = torch.tensor(mix.shape[0]*mix.shape[1]*mix.shape[2]).cuda().float()

  model.hidden = model.init_hidden(len(mix))
  embeddings = model(mix) # shape: (batch, seq_length, feat_dim, emb_dim)
#  print("emb[0,0]:", embeddings[0,0,0,:].detach().cpu().numpy())
#  print("emb[0,1]:", embeddings[0,0,1,:].detach().cpu().numpy())
#  print("emb[1,0]:", embeddings[0,1,0,:].detach().cpu().numpy())
#  print("emb[1,1]:", embeddings[0,1,1,:].detach().cpu().numpy())
  dims = embeddings.shape
  embeddings = embeddings.view((dims[0], dims[1]*dims[2], dims[3])) # shape: (batch, seq_length*feat_dim, emb_dim)
#  print("res[0,0]:", embeddings[0,0,:].detach().cpu().numpy())
#  print("res[0,1]:", embeddings[0,1,:].detach().cpu().numpy())
#  print("res[1,0]:", embeddings[0,129,:].detach().cpu().numpy())
#  print("res[1,1]:", embeddings[0,130,:].detach().cpu().numpy())
  affinity = affinity.view((dims[0], dims[1]*dims[2], -1)) # shape: (batch, seq_length*feat_dim, num_spk)
#  print("res[0,0]:", affinity[0,0,:].detach().cpu().numpy())
#  print("res[0,1]:", affinity[0,1,:].detach().cpu().numpy())
#  print("res[1,0]:", affinity[0,129,:].detach().cpu().numpy())
#  print("res[1,1]:", affinity[0,130,:].detach().cpu().numpy())

  if plotdir:
    os.system("mkdir -p "+plotdir)
    prefix = plotdir+'/'
    sub_aff_mat = torch.matmul(affinity[0,0:50,:], affinity.permute(0,2,1)[0,:,0:50]).detach().cpu().numpy()
    sub_emb_mat = torch.matmul(embeddings[0,0:50,:], embeddings.permute(0,2,1)[0,:,0:50]).detach().cpu().numpy()
    plot.plot_spec(sub_aff_mat, prefix+"Sub_Affinity_Matrix.png")
    plot.plot_spec(sub_emb_mat, prefix+"Sub_Estimated_Affinity_Matrix.png")
    plot.plot_spec(mix[0].detach().cpu().numpy(), prefix+'Input.png')
#    plot.clust_plot(embeddings[0].detach().cpu().numpy(), mix[0].shape[::-1], prefix+'Cluster_Partitioning.png')
    plot.clust_plot(embeddings[0].detach().cpu().numpy(), mix[0].shape, prefix+'Cluster_Partitioning.png')
    for i in range(affinity.shape[2]):
      plot.plot_spec(affinity[0,:,i].view(mix[0].shape).detach().cpu().numpy(), prefix+'Source'+str(i+1)+'_Bins.png')
    plot.plot_embs(embeddings[0:5].detach().cpu().numpy(), affinity[0:5].detach().cpu().numpy(), prefix+'Embeddings.png')
    # (seq_length*feat_dim, emb_dim) (seq_length*feat_dim, num_spk)

  loss += frob_sq_ata(embeddings.permute(0,2,1), embeddings)
  loss -= 2*frob_sq_ata(embeddings.permute(0,2,1), affinity)
#  loss += frob_sq_ata(affinity.permute(0,2,1), affinity) # Doesn't depend on network
#  length = embeddings.shape[0]
#  print("length:", length)
#  loss += frob_sq_ata(embeddings.permute(0,2,1)[0:length,:,:], embeddings[0:length,:,:])
#  loss -= 2*frob_sq_ata(embeddings.permute(0,2,1)[0:length,:,:], affinity[0:length,:,:])


#  with torch.no_grad():
#    reg_loss = torch.sum((torch.matmul(embeddings.detach().cpu()[0,:,:], embeddings.detach().cpu().permute(0,2,1)[0,:,:])-torch.matmul(affinity.detach().cpu()[0,:,:], affinity.detach().cpu().permute(0,2,1)[0,:,:])).pow(2))
#    red_loss = frob_sq_ata(embeddings.detach().cpu().permute(0,2,1)[0,:,:], embeddings.detach().cpu()[0,:,:])-2*frob_sq_ata(embeddings.detach().cpu().permute(0,2,1)[0,:,:], affinity.detach().cpu()[0,:,:])+frob_sq_ata(affinity.detach().cpu().permute(0,2,1)[0,:,:], affinity.detach().cpu()[0,:,:])
#    print("reg_loss:", reg_loss.numpy(), "red_loss:", red_loss.numpy())
#    print("source1 count:", torch.sum(affinity.detach().cpu()[0,:,0]).numpy())
#    print("source2 count:", torch.sum(affinity.detach().cpu()[0,:,1]).numpy())
#    print("frob_sq_ata YTY:", frob_sq_ata(affinity.detach().cpu().permute(0,2,1)[0,:,:], affinity.detach().cpu()[0,:,:]).numpy())

  return loss/norm, norm

def frob_sq_ata(mat1, mat2):
  product = torch.matmul(mat1, mat2)
  square = product.pow(2)
  return torch.sum(square)

# define test pass
def compute_masks(model, batch_sample, out_dir):
  model.zero_grad()

  mix = batch_sample['mix'].cuda()
  num_spk = batch_sample['num_spk']
  name = batch_sample['name']

  batch = len(name)

  model.hidden = model.init_hidden(len(mix))
  embeddings = model(mix) # shape: (batch, seq_length, feat_dim, emb_dim)

  embeddings = embeddings.view((embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2], embeddings.shape[3]))
  for i in range(batch):
    dict_out = dict()
    kmeans = cluster.KMeans(n_clusters=num_spk[i])
    assigns = kmeans.fit_predict(embeddings[i].detach().cpu().numpy()).reshape(mix[0].shape)
    for source in range(num_spk[i]):
      dict_out['s'+str(source+1)] = (assigns == source).astype(float).T
    np.savez_compressed(out_dir+'/'+name[i], **dict_out)
