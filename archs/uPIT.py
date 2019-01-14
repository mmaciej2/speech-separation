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

# Define dataset
class TrainSet(Dataset):

  def __init__(self, filelist, location=""):
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
    mix_mag_spec = np.load(self.list[idx])['mix'].transpose()
    source_mag_spec_1 = np.load(mix_name_to_source_mag_spec_name(self.list[idx], 1))['s1'].transpose()
    source_mag_spec_2 = np.load(mix_name_to_source_mag_spec_name(self.list[idx], 2))['s2'].transpose()

    permute_1 = np.concatenate((source_mag_spec_1, source_mag_spec_2), axis=1)
    permute_2 = np.concatenate((source_mag_spec_2, source_mag_spec_1), axis=1)

    sample = {'mix': mix_mag_spec, 'permute1': permute_1, 'permute2': permute_2}
    return sample

class TestSet(Dataset):

  def __init__(self, filelist):
    self.list = [line.rstrip('\n').split(' ')[1] for line in open(filelist)]
    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.abs(np.load(self.list[idx])['mix']).transpose()

    sample = {'mix': mix_mag_spec, 'name': os.path.basename(self.list[idx])}
    return sample

# define nnet
class EnhBLSTM(nn.Module):
  def __init__(self, gpuid):
    super(EnhBLSTM, self).__init__()

    self.gpuid = gpuid

    self.blstm = nn.LSTM(257, 600, num_layers=2, bidirectional=True)

    self.lin = nn.Linear(600*2, 514)

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

  mix = batch_sample['mix'].cuda()
  permute1 = batch_sample['permute1'].cuda()
  permute2 = batch_sample['permute2'].cuda()
  batch = int(mix.batch_sizes[0])

  model.zero_grad()
  model.hidden = model.init_hidden(batch)

  mask_out = model(mix)

  mixes, lens = pad_packed_sequence(mix, batch_first=True)
  permutations1, lens = pad_packed_sequence(permute1, batch_first=True)
  permutations2, lens = pad_packed_sequence(permute2, batch_first=True)
  lengths = lens.float().cuda()
  double_mix = torch.cat((mixes, mixes), dim=2)
  masked = mask_out * double_mix
  loss1 = torch.sum(loss_function(masked, permutations1).view(batch, -1), dim=1)
  loss2 = torch.sum(loss_function(masked, permutations2).view(batch, -1), dim=1)

  if plotdir:
    os.system("mkdir -p "+plotdir)
    qp.plot(mixes[0].detach().cpu().numpy(), plotdir+'/mix.png')
    qp.plot(masked[0].detach().cpu().numpy(), plotdir+'/masked_mix.png')
    if loss1[0] < loss2[0]:
      qp.plot(permutations1[0].detach().cpu().numpy(), plotdir+'/chosen_permutation.png')
    else:
      qp.plot(permutations2[0].detach().cpu().numpy(), plotdir+'/chosen_permutation.png')

  return torch.mean(torch.min(loss1, loss2)/lengths/514)

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
    file_dict['s1'] = mask[0:257]
    file_dict['s2'] = mask[257:514]
    np.savez_compressed(out_dir+'/'+name[i], **file_dict)
