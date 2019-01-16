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

  def collate_sub_batch(self, batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], collections.Mapping):
      sort_inds = np.argsort(np.array([len(d[self.key]) for d in batch]))[::-1]
      return {key: self.collate_sub_batch([batch[i][key] for i in sort_inds]) for key in batch[0]}
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ == 'ndarray':
      # array of string classes and object
      if re.search('[SaUO]', batch[0].dtype.str) is not None:
        raise TypeError(error_msg.format(elem.dtype))
      return pack_sequence([(torch.from_numpy(b)).float() for b in batch])
    else:
      return default_collate(batch)

  def __call__(self, batch):
    if not self.key:
      return default_collate(batch)
    else:
      sys.stdout.flush()
      batch_out = []
      if "num_spk" in batch[0].keys():
        max_spk = max([int(d["num_spk"]) for d in batch])
        for num_spk in range(max_spk+1):
          inds = [i for i in range(len(batch)) if int(batch[i]["num_spk"]) == num_spk]
          if len(inds) > 0:
            batch_out.append(self.collate_sub_batch([batch[i] for i in inds]))
          else:
            batch_out.append({})
      else:
        max_spk = max([len(d.keys())-1 for d in batch])
        for num_spk in range(max_spk+1):
          inds = [i for i in range(len(batch)) if len(batch[i].keys())-1 == num_spk]
          if len(inds) > 0:
            batch_out.append(self.collate_sub_batch([batch[i] for i in inds]))
          else:
            batch_out.append({})
      return batch_out

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
    self.collator = Collator('combo')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    feat = np.load(self.list[idx])
    mix_mag_spec = feat['mix'].transpose()
    atten_mask = np.ones(mix_mag_spec.shape)
    combo = np.concatenate((mix_mag_spec, atten_mask), axis=1)

    sample = {'combo': combo}

    for src in range(len(feat.files)-1):
      source_mag_spec = feat['s'+str(src+1)].transpose()
      sample["source"+str(src+1)] = source_mag_spec

    return sample

class TestSet(Dataset):

  def __init__(self, datadir):
    self.list = [line.rstrip('\n').split(' ')[1] for line in open(datadir+"/feats_test.scp")]
    self.num_spks = [int(line.rstrip('\n').split(' ')[1]) for line in open(datadir+"/utt2num_spk")]
    self.collator = Collator('combo')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.abs(np.load(self.list[idx])['mix']).transpose()
    atten_mask = np.ones(mix_mag_spec.shape)
    combo = np.concatenate((mix_mag_spec, atten_mask), axis=1)

    sample = {'combo': combo, 'name': os.path.basename(self.list[idx]), 'num_spk': self.num_spks[idx]}
    return sample

# define nnet
class EnhBLSTM(nn.Module):
  def __init__(self, gpuid):
    super(EnhBLSTM, self).__init__()

    self.gpuid = gpuid

    self.blstm = nn.LSTM(514, 600, num_layers=2, bidirectional=True)

    self.lin = nn.Linear(600*2, 257)

    self.bn = nn.BatchNorm1d(600*2)

  def init_hidden(self, batch_size):
    if self.gpuid > -1:
      return (torch.randn(2*2, batch_size, 600).cuda(),
              torch.randn(2*2, batch_size, 600).cuda())
    else:
      return (torch.randn(2*2, batch_size, 600),
              torch.randn(2*2, batch_size, 600))

  def forward(self, x):
    # x: packed sequence of dim 514

    x, self.hidden = self.blstm(x, self.hidden)
    # x: packed sequence of dim 600*2

    x, lens = pad_packed_sequence(x, batch_first=True)
    # x: tensor of shape (batch, seq_length, 600*2)

    x = self.bn(x.permute(0,2,1).contiguous()).permute(0,2,1)
    # x: tensor of shape (batch, seq_length, 600*2)

    x = self.lin(x)
    # x: tensor of shape (batch, seq_length, 257)

    x = F.sigmoid(x)
    # x: tensor of shape (batch, seq_length, 257)

    return x

# define training pass
def compute_loss(model, batch_sample, plotdir=""):
  loss_function = nn.MSELoss(reduce=False)

  model.zero_grad()
  loss = 0

  for num_spk in range(len(batch_sample)):
    if len(batch_sample[num_spk]) > 0:
      combo = batch_sample[num_spk]['combo'].cuda()
      batch = int(combo.batch_sizes[0])

      model.hidden = model.init_hidden(batch)

      sources = []
      for i in range(num_spk):
        source = batch_sample[num_spk]['source'+str(i+1)].cuda()
        source, lens = pad_packed_sequence(source, batch_first=True)
        # source: tensor of shape (batch, seq_length, 257)
        sources.append(source)
      source_usage = [[] for _ in range(num_spk)]
      for dnn_pass in range(num_spk):
        mask_out = model(combo)
        # mask_out: tensor of shape (batch, seq_length, 257)

        combos, lens = pad_packed_sequence(combo, batch_first=True)
        # combos: tensor of shape (batch, seq_length, 514)
        mixes = torch.index_select(combos, 2, torch.LongTensor(range(257)).cuda())
        # mixes: tensor of shape (batch, seq_length, 257)
        lengths = lens.float().cuda()

        masked = mask_out * mixes
        losses = torch.stack([torch.sum(loss_function(masked, source).view(batch, -1), dim=1) for source in sources])

        for source_ind in range(num_spk):
          for index in source_usage[source_ind]:
            losses[source_ind][index] = float("Inf")

        min_losses, indices = torch.min(losses, 0)
        for sample_ind in range(batch):
          source_usage[indices[sample_ind]].append(sample_ind)

        loss += torch.mean(min_losses/lengths/257/batch)

        if plotdir:
          os.system("mkdir -p "+plotdir)
          if dnn_pass == 0:
            plot.plot_spec(combos[0].detach().cpu().numpy()[:,0:257], plotdir+'/'+str(num_spk)+'-Spk_Mix.png')
          prefix = plotdir+'/'+str(num_spk)+'-Spk_Pass-'+str(dnn_pass+1)+'_'
          plot.plot_spec(combos[0].detach().cpu().numpy(), prefix+'Input.png')
          plot.plot_spec(combos[0].detach().cpu().numpy()[:,258:514], prefix+'Attenmask.png')
          plot.plot_spec(mask_out[0].detach().cpu().numpy(), prefix+'Mask_Out.png')
          plot.plot_spec(masked[0].detach().cpu().numpy(), prefix+'Masked_Mix.png')
          plot.plot_spec(sources[indices[0]][0].detach().cpu().numpy(), prefix+'Chosen_Source.png')

        spec_zeros = torch.zeros(mask_out.shape).cuda()
        residual_comp = torch.cat((spec_zeros, mask_out), 2)
        combos = F.relu_(combos - residual_comp)
        combo = pack_padded_sequence(combos, lens, batch_first=True)

  return loss

# define test pass
def compute_masks(model, batch_sample, out_dir):
  model.zero_grad()

  for num_spk in range(len(batch_sample)):
    if len(batch_sample[num_spk]) > 0:
      combo = batch_sample[num_spk]['combo'].cuda()
      name = batch_sample[num_spk]['name']
      batch = int(combo.batch_sizes[0])

      model.hidden = model.init_hidden(batch)

      dicts = [dict() for _ in range(batch)]
      for dnn_pass in range(num_spk):
        mask_out = model(combo)

        combos, lens = pad_packed_sequence(combo, batch_first=True)
        spec_zeros = torch.zeros(mask_out.shape).cuda()
        residual_comp = torch.cat((spec_zeros, mask_out), 2)
        combos = combos - residual_comp
        combo = pack_padded_sequence(combos, lens, batch_first=True)

        for i in range(len(name)):
          dicts[i]['s'+str(dnn_pass+1)] = mask_out[i].cpu().numpy().transpose()[:,0:lens[i]]

      for i in range(batch):
        np.savez_compressed(out_dir+'/'+name[i], **(dicts[i]))
