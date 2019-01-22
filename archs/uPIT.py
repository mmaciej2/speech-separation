#/usr/bin/env python3

import torch
import sys
import os
import numpy as np
import collections
import re
import itertools

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
    filelist = datadir+"/feats_train.scp"
    if location:
      with open(filelist) as F:
        indir = os.path.dirname(F.readline().rstrip().split(' ')[1])+'/'
      print("rsync -r --bwlimit=10000 "+indir+" "+location)
      os.system("rsync -r --bwlimit=10000 "+indir+" "+location)
      self.list = [location+'/'+line.split(' ')[0]+'.npz' for line in open(filelist)]
    else:
      self.list = [line.rstrip('\n').split(' ')[1] for line in open(filelist)]
    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    feat = np.load(self.list[idx])
    mix_mag_spec = feat['mix'].transpose()

    sample = {'mix': mix_mag_spec}

    for src in range(len(feat.files)-1):
      source_mag_spec = feat['s'+str(src+1)].transpose()
      sample["source"+str(src+1)] = source_mag_spec

    return sample

class TestSet(Dataset):

  def __init__(self, datadir):
    self.list = [line.rstrip('\n').split(' ')[1] for line in open(datadir+"/feats_test.scp")]
    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.abs(np.load(self.list[idx])['mix']).transpose()

    sample = {'mix': mix_mag_spec, 'name': os.path.basename(self.list[idx])}
    return sample

# define nnet
class SepDNN(nn.Module):
  def __init__(self, gpuid, **kwargs):
    super(SepDNN, self).__init__()

    self.gpuid = gpuid

    if 'feat_dim' in kwargs.keys():
      self.feat_dim = int(kwargs['feat_dim'])
    else:
      self.feat_dim = 257
    if 'num_spk' in kwargs.keys():
      self.num_spk = int(kwargs['num_spk'])
    else:
      self.num_spk = 2

    for key in kwargs.keys():
      print('modelparam:', key, kwargs[key])

    self.blstm = nn.LSTM(self.feat_dim, 600, num_layers=2, bidirectional=True)

    self.lin = nn.Linear(600*2, self.feat_dim*self.num_spk)

    self.bn = nn.BatchNorm1d(600*2)

  def init_hidden(self, batch_size):
    if self.gpuid > -1:
      return (torch.randn(2*2, batch_size, 600).cuda(),
              torch.randn(2*2, batch_size, 600).cuda())
    else:
      return (torch.randn(2*2, batch_size, 600),
              torch.randn(2*2, batch_size, 600))

  def forward(self, x):
    # x: packed sequence of dim feat_dim

    x, self.hidden = self.blstm(x, self.hidden)
    # x: packed sequence of dim 600*2

    x, lens = pad_packed_sequence(x, batch_first=True)
    # x: tensor of shape (batch, seq_length, 600*2)

    x = self.bn(x.permute(0,2,1).contiguous()).permute(0,2,1)
    # x: tensor of shape (batch, seq_length, 600*2)

    x = self.lin(x)
    # x: tensor of shape (batch, seq_length, feat_dim*num_spk)

    x = F.sigmoid(x)
    # x: tensor of shape (batch, seq_length, feat_dim*num_spk)

    return x

def compute_cv_loss(model, batch_sample, plotdir=""):
  if plotdir:
    loss, norm = compute_loss(model, batch_sample, plotdir)
  else:
    loss, norm = compute_loss(model, batch_sample)
  return loss, norm

# define training pass
def compute_loss(model, batch_sample, plotdir=""):
  loss_function = nn.MSELoss(reduce=False)

  mix = batch_sample['mix'].cuda()
  batch = int(mix.batch_sizes[0])
  sources = []
  for i in range(model.num_spk):
    source = batch_sample['source'+str(i+1)].cuda()
    source, lens = pad_packed_sequence(source, batch_first=True)
    # source: tensor of shape (batch, seq_length, feat_dim)
    sources.append(source)

  model.zero_grad()
  model.hidden = model.init_hidden(batch)

  loss = 0
  norm = 0

  mask_out = model(mix)
  # mask_out: tensor of shape (batch, seq_length, feat_dim*2)

  mixes, lens = pad_packed_sequence(mix, batch_first=True)
  lengths = lens.float().cuda()
  # mixes: tensor of shape (batch, seq_length, feat_dim)
  stacked_mix = torch.cat([mixes for _ in range(model.num_spk)], dim=2)
  # stacked_mix: tensor of shape (batch, seq_length, feat_dim*num_spk)
  masked = mask_out * stacked_mix
  # masked: tensor of shape (batch, seq_length, feat_dim*num_spk)

  perms = list(itertools.permutations(range(model.num_spk)))
#  losses = []
#  for perm_i in range(len(perms)):
#    out_perms = torch.cat([sources[i] for i in perms[perm_i]], dim=2)
#    losses.append(torch.sum(loss_function(masked, out_perms).view(batch, -1), dim=1))
#  losses = torch.stack(losses)
  losses = torch.stack([torch.sum(loss_function(masked, permute).view(batch, -1), dim=1) for permute in [torch.cat([sources[i] for i in perm], dim=2) for perm in perms]])

  min_losses, indices = torch.min(losses, 0)

  loss += torch.sum(min_losses)/model.num_spk
  norm += torch.sum(lengths)*model.feat_dim

  if plotdir:
    os.system("mkdir -p "+plotdir)
    plot.plot_spec(mixes[0].detach().cpu().numpy(), plotdir+'/Mixture.png')
    plot.plot_spec(masked[0].detach().cpu().numpy(), plotdir+'/Masked_Mixture.png')
    permutation = perms[indices[0]]
    plot.plot_spec(torch.cat([sources[i][0] for i in permutation], dim=1).detach().cpu().numpy(), plotdir+'/Chosen_Permutation.png')

  return loss/norm, norm

# define test pass
def compute_masks(model, batch_sample, out_dir):
  mix = batch_sample['mix'].cuda()
  name = batch_sample['name']
  batch = int(mix.batch_sizes[0])

  model.zero_grad()
  model.hidden = model.init_hidden(batch)

  mask_out = model(mix)
  lens = pad_packed_sequence(mix, batch_first=True)[1]

  for i in range(len(name)):
    mask = mask_out[i].cpu().numpy().transpose()[:,0:lens[i]]
    file_dict = dict()
    for src in range(model.num_spk):
      file_dict['s'+str(src+1)] = mask[src*model.feat_dim:(src+1)*model.feat_dim]
    np.savez_compressed(out_dir+'/'+name[i], **file_dict)
