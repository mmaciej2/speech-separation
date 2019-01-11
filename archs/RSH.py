#/usr/bin/env python3

import torch
import sys
import os
import numpy as np
import collections
import re

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate, numpy_type_map
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import quick_plot as qp


# finds filename for mask corresponding to a mix spec
def mix_name_to_source_mag_spec_name(filename, index):
  split = os.path.basename(filename).split('-')
  if index == 1:
    return filename.replace("/mix/","/s1/")
  else:
    return filename.replace("/mix/","/s2/")

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

# Define datasets
class TrainSet2mix(Dataset):

  def __init__(self, filelist, location=""):
    if location:
      with open(filelist) as F:
        indir = F.readline().split('/mix/')[0]+'/'
      print("rsync -r --bwlimit=10000 "+indir+" "+location)
      os.system("rsync -r --bwlimit=10000 "+indir+" "+location)
      self.list = [location+'/mix/'+line.rstrip('\n').split('/mix/')[1] for line in open(filelist)]
    else:
      self.list = [line.rstrip('\n') for line in open(filelist)]
    self.collator = Collator('combo')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.load(self.list[idx]).transpose()
    atten_mask = np.ones(mix_mag_spec.shape)
    combo = np.concatenate((mix_mag_spec, atten_mask), axis=1)

    source_mag_spec_1 = np.load(mix_name_to_source_mag_spec_name(self.list[idx], 1)).transpose()
    source_mag_spec_2 = np.load(mix_name_to_source_mag_spec_name(self.list[idx], 2)).transpose()

    sample = {'combo': combo, 'source1': source_mag_spec_1, 'source2': source_mag_spec_2}
    return sample

class TestSet2mix(Dataset):

  def __init__(self, filelist):
    self.list = [line.rstrip('\n') for line in open(filelist)]
    self.collator = Collator('combo')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.load(self.list[idx]).transpose()
    atten_mask = np.ones(mix_mag_spec.shape)
    combo = np.concatenate((mix_mag_spec, atten_mask), axis=1)

    sample = {'combo': combo, 'name': os.path.basename(self.list[idx])}
    return sample

# define nnet
class EnhBLSTM(nn.Module):
  def __init__(self, gpuid):
    super(EnhBLSTM, self).__init__()

    self.gpuid = gpuid

    self.blstm = nn.LSTM(514, 600, num_layers=2, bidirectional=True)

    self.lin = nn.Linear(600*2, 257)

    self.bn2 = nn.BatchNorm1d(600*2)

  def init_hidden(self, batch_size):
    if self.gpuid > -1:
      return (torch.randn(2*2, batch_size, 600).cuda(),
              torch.randn(2*2, batch_size, 600).cuda())
    else:
      return (torch.randn(2*2, batch_size, 600),
              torch.randn(2*2, batch_size, 600))

  def forward(self, x):
    x, self.hidden = self.blstm(x, self.hidden)
    x, lens = pad_packed_sequence(x, batch_first=True)
    x = self.bn2(x.permute(0,2,1).contiguous()).permute(0,2,1)
    x = self.lin(x)
    x = F.sigmoid(x)
    return x

# define training pass
def compute_loss(model, batch_sample, plotdir=""):
  loss_function = nn.MSELoss(reduce=False)

  combo = batch_sample['combo'].cuda()
  source1 = batch_sample['source1'].cuda()
  source2 = batch_sample['source2'].cuda()
  batch = int(combo.batch_sizes[0])

  model.zero_grad()
  model.hidden = model.init_hidden(batch)

  loss = 0
  num_sources = 2
  source_usage = [[] for _ in range(num_sources)]
  for dnn_pass in range(num_sources):
    mask_out = model(combo)

    combos, lens = pad_packed_sequence(combo, batch_first=True)
    mixes = torch.index_select(combos, 2, torch.LongTensor(range(257)).cuda())
    source1s, lens = pad_packed_sequence(source1, batch_first=True)
    source2s, lens = pad_packed_sequence(source2, batch_first=True)
    lengths = lens.float().cuda()

    masked = mask_out * mixes
    loss1 = torch.sum(loss_function(masked, source1s).view(batch, -1), dim=1)
    loss2 = torch.sum(loss_function(masked, source2s).view(batch, -1), dim=1)
    losses = torch.stack((loss1, loss2))
    for source_ind in range(num_sources):
      for index in source_usage[source_ind]:
        losses[source_ind][index] = float("Inf")
    min_losses, indices = torch.min(losses, 0)
    for sample_ind in range(batch):
      source_usage[indices[sample_ind]].append(sample_ind)
    loss += torch.mean(min_losses/lengths/257/batch)

    if plotdir:
      os.system("mkdir -p "+plotdir)
      prefix = plotdir+'/pass'+str(dnn_pass)+'_'
      qp.plot(combos[0].detach().cpu().numpy(), prefix+'input.png')
      qp.plot(combos[0].detach().cpu().numpy()[:,258:514], prefix+'attenmask.png')
      qp.plot(mask_out[0].detach().cpu().numpy(), prefix+'mask_out.png')
      if indices[0] == 0:
        qp.plot(source1s[0].detach().cpu().numpy(), prefix+'source.png')
      else:
        qp.plot(source2s[0].detach().cpu().numpy(), prefix+'source.png')

    spec_zeros = torch.zeros(mask_out.shape).cuda()
    residual_comp = torch.cat((spec_zeros, mask_out), 2)
    combos = combos - residual_comp
    combo = pack_padded_sequence(combos, lens, batch_first=True)

  return loss

# define test pass
def compute_masks(model, batch_sample, out_dir):
  combo = batch_sample['combo'].cuda()
  name = batch_sample['name']
  batch = int(combo.batch_sizes[0])

  model.zero_grad()
  model.hidden = model.init_hidden(batch)

  num_sources = 2
  for dnn_pass in range(num_sources):
    mask_out = model(combo)

    combos, lens = pad_packed_sequence(combo, batch_first=True)
    spec_zeros = torch.zeros(mask_out.shape).cuda()
    residual_comp = torch.cat((spec_zeros, mask_out), 2)
    combos = combos - residual_comp
    combo = pack_padded_sequence(combos, lens, batch_first=True)

    for i in range(len(name)):
      mask = mask_out[i].cpu().numpy().transpose()[:,0:lens[i]]
      np.save(os.path.join(out_dir, 's'+str(dnn_pass+1), name[i]), mask)
