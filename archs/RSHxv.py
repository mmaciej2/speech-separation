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

eps = 1e-10


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
    elif isinstance(batch[0], list):
      return batch
    else:
      return default_collate(batch)

  def __call__(self, batch):
    if not self.key:
      return default_collate(batch)
    else:
      if "num_spk" in batch[0].keys():
        max_spk = max([int(d["num_spk"]) for d in batch])
        batch_out = MultiSpkBatch(max_spk+1, len(batch))
        for num_spk in range(max_spk+1):
          inds = [i for i in range(len(batch)) if int(batch[i]["num_spk"]) == num_spk]
          batch_out.sub_batch_lens.append(len(inds))
          if len(inds) > 0:
            batch_out.append(self.collate_sub_batch([batch[i] for i in inds]))
          else:
            batch_out.append({})
      else:
        max_spk = max([len(d.keys())-1 for d in batch])
        batch_out = MultiSpkBatch(max_spk+1, len(batch))
        for num_spk in range(max_spk+1):
          inds = [i for i in range(len(batch)) if len(batch[i].keys())-1 == num_spk]
          batch_out.sub_batch_lens.append(len(inds))
          if len(inds) > 0:
            batch_out.append(self.collate_sub_batch([batch[i] for i in inds]))
          else:
            batch_out.append({})
      return batch_out

class MultiSpkBatch():

  def __init__(self, max_spk, length):
    self.sub_batches = list()
    self.sub_batch_lens = list()
    self.max_spk = max_spk
    self.length = length

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return self.sub_batches[idx]

  def append(self, elem):
    self.sub_batches.append(elem)

# Define dataset
class TrainSet(Dataset):

  def __init__(self, datadir, location=""):
    self.num_spks = [int(line.rstrip('\n').split(' ')[1]) for line in open(datadir+"/utt2num_spk")]

    self.src_ids = []
    spk_id_no = 0
    spk_dict = dict()
    for line in open(datadir+"/utt2spk"):
      sess_spks = []
      for spk in [line.rstrip().split(' ')[i] for i in range(1, len(line.split(' ')))]:
        if spk in spk_dict.keys():
          sess_spks.append(spk_dict[spk])
        else:
          sess_spks.append(spk_id_no)
          spk_dict[spk] = spk_id_no
          spk_id_no = spk_id_no + 1
      self.src_ids.append(sess_spks)

    filelist = datadir+"/feats_train.scp"
    if location:
      print("tools/copy_scp_data_to_dir.sh "+filelist+" "+location+" --bwlimit 10000")
      os.system("tools/copy_scp_data_to_dir.sh "+filelist+" "+location+" --bwlimit 10000")
      self.list = [location+'/'+line.rstrip('\n').split(' ')[1] for line in open(filelist)]
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

    sample = {'combo': combo, 'num_spk': self.num_spks[idx], 'spk_ids': self.src_ids[idx]}

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
class SepDNN(nn.Module):
  def __init__(self, gpuid, **kwargs):
    super(SepDNN, self).__init__()

    self.gpuid = gpuid

    num_spk = int(kwargs['num_spk'])

    if 'feat_dim' in kwargs.keys():
      self.feat_dim = int(kwargs['feat_dim'])
    else:
      self.feat_dim = 257

    if 'alpha' in kwargs.keys():
      self.alpha = float(kwargs['alpha'])
    else:
      self.alpha = 1.0

    for key in kwargs.keys():
      print('modelparam:', key, kwargs[key])

    # Base LSTM
    self.blstm = nn.LSTM(self.feat_dim*2, 600, num_layers=2, bidirectional=True)
    self.bn1 = nn.BatchNorm1d(600*2)

    # Separation output
    self.lin = nn.Linear(600*2, self.feat_dim)

    # Xvector segment-level
    self.xv_lin1 = nn.Linear(600*2*2, 512)
    self.bn2 = nn.BatchNorm1d(512)
    self.xv_lin2 = nn.Linear(512, 200)
    self.bn3 = nn.BatchNorm1d(200)
    self.xv_lin3 = nn.Linear(200, num_spk)


  def init_hidden(self, batch_size):
    if self.gpuid > -1:
      return (torch.randn(2*2, batch_size, 600).cuda(),
              torch.randn(2*2, batch_size, 600).cuda())
    else:
      return (torch.randn(2*2, batch_size, 600),
              torch.randn(2*2, batch_size, 600))

  def forward(self, x): # x: packed seq dim feat_dim*2
    x, self.hidden = self.blstm(x, self.hidden) # x: packed seq dim 600*2
    x, lens = pad_packed_sequence(x, batch_first=True) # x: (batch, seq_len, 600*2)
    x = self.bn1(x.permute(0,2,1).contiguous()).permute(0,2,1) # x: (batch, seq_len, 600*2)

    ### Embedding branch
    y = torch.cat((torch.mean(x, 1),
                   torch.sqrt(torch.mean(torch.pow(x - torch.mean(x,1).unsqueeze(1).expand(x.shape), 2), 1)+eps)),
                  dim=1) # y: (batch, 600*2*2)
    y = self.xv_lin1(y) # y: (batch, 512)
    y = self.bn2(y)
    y = self.xv_lin2(y) # y: (batch, 200)
    y = self.bn3(y)
    y = self.xv_lin3(y) # y: (batch, num_spk)

    ### Separation branch
    x = self.lin(x) # x: (batch, seq_len, feat_dim)
    x = torch.sigmoid(x) # x: (batch, seq_len, feat_dim)

    return x, y

  def forward_separate(self, x): # x: packed seq dim feat_dim*2
    x, self.hidden = self.blstm(x, self.hidden) # x: packed seq dim 600*2
    x, lens = pad_packed_sequence(x, batch_first=True) # x: (batch, seq_len, 600*2)
    x = self.bn1(x.permute(0,2,1).contiguous()).permute(0,2,1) # x: (batch, seq_len, 600*2)
    x = self.lin(x) # x: (batch, seq_len, feat_dim)
    x = torch.sigmoid(x) # x: (batch, seq_len, feat_dim)
    return x

  def forward_embedding(self, x, layer_id): # x: packed seq dim feat_dim*2
    x, self.hidden = self.blstm(x, self.hidden) # x: packed seq dim 600*2
    x, lens = pad_packed_sequence(x, batch_first=True) # x: (batch, seq_len, 600*2)
    x = self.bn(x.permute(0,2,1).contiguous()).permute(0,2,1) # x: (batch, seq_len, 600*2)
    x = torch.cat((torch.mean(x, 1), torch.std(x, 1)), dim=1) # x: (batch, 600*2*2)
    x = self.xv_lin1(x) # x: (batch, 512)
    if layer_id == 0:
     return x # (batch, 512)
    else:
      return self.xv_lin2(x) # (batch, 200)
    

# define training pass
def compute_loss(model, batch_sample, plotdir=""):
  sep_loss_fn = nn.MSELoss(reduce=False)
  sid_loss_fn = nn.CrossEntropyLoss(reduce=True)

  model.zero_grad()

  sep_loss = 0
  sid_loss = 0
  norm = 0

  for num_spk in range(batch_sample.max_spk):
    if len(batch_sample[num_spk]) > 0:
      batch = batch_sample.sub_batch_lens[num_spk]
      combo = batch_sample[num_spk]['combo'].cuda()

      model.hidden = model.init_hidden(batch)

      sources = []
      for i in range(num_spk):
        source = batch_sample[num_spk]['source'+str(i+1)].cuda()
        source, lens = pad_packed_sequence(source, batch_first=True)
        # source: tensor of shape (batch, seq_length, feat_dim)
        sources.append(source)
      source_usage = [[] for _ in range(num_spk)]
      for dnn_pass in range(num_spk):
        mask_out, id_vec = model(combo)
        # mask_out: tensor of shape (batch, seq_length, feat_dim)

        combos, lens = pad_packed_sequence(combo, batch_first=True)
        # combos: tensor of shape (batch, seq_length, feat_dim*2)
        mixes = torch.index_select(combos, 2, torch.LongTensor(range(model.feat_dim)).cuda())
        # mixes: tensor of shape (batch, seq_length, feat_dim)
        lengths = lens.float().cuda()

        masked = mask_out * mixes
        losses = torch.stack([torch.sum(sep_loss_fn(masked, source).view(batch, -1), dim=1) for source in sources])

        for source_ind in range(num_spk):
          for index in source_usage[source_ind]:
            losses[source_ind][index] = float("Inf")

        min_losses, indices = torch.min(losses, 0)
        for sample_ind in range(batch):
          source_usage[indices[sample_ind]].append(sample_ind)

        ids = torch.empty((batch), dtype=torch.long).cuda()
        for sample_ind in range(batch):
          ids[sample_ind] = batch_sample[num_spk]['spk_ids'][sample_ind][indices[sample_ind]]

        sid_loss += sid_loss_fn(id_vec, ids)/num_spk
        sep_loss += torch.sum(min_losses)/num_spk
        norm += torch.sum(lengths)*model.feat_dim

        if plotdir:
          os.system("mkdir -p "+plotdir)
          if dnn_pass == 0:
            plot.plot_spec(combos[0].detach().cpu().numpy()[:,0:model.feat_dim], plotdir+'/'+str(num_spk)+'-Spk_Mix.png')
          prefix = plotdir+'/'+str(num_spk)+'-Spk_Pass-'+str(dnn_pass+1)+'_'
          plot.plot_spec(combos[0].detach().cpu().numpy(), prefix+'Input.png')
          plot.plot_spec(combos[0].detach().cpu().numpy()[:,model.feat_dim+1:model.feat_dim*2], prefix+'Attenmask.png')
          plot.plot_spec(mask_out[0].detach().cpu().numpy(), prefix+'Mask_Out.png')
          plot.plot_spec(masked[0].detach().cpu().numpy(), prefix+'Masked_Mix.png')
          plot.plot_spec(sources[indices[0]][0].detach().cpu().numpy(), prefix+'Chosen_Source.png')

        spec_zeros = torch.zeros(mask_out.shape).cuda()
        residual_comp = torch.cat((spec_zeros, mask_out), 2)
        combos = F.relu_(combos - residual_comp)
        combo = pack_padded_sequence(combos, lens, batch_first=True)

  print("sep_loss:", (sep_loss/norm).detach().cpu().numpy(), "sid_loss:", sid_loss.detach().cpu().numpy(), "alpha:", model.alpha)
  loss = sep_loss/norm + model.alpha*sid_loss, norm
  return loss

# define cv loss computation
def compute_cv_loss(model, batch_sample, plotdir=""):
  sep_loss_fn = nn.MSELoss(reduce=False)

  model.zero_grad()

  loss = 0
  norm = 0

  for num_spk in range(batch_sample.max_spk):
    if len(batch_sample[num_spk]) > 0:
      batch = batch_sample.sub_batch_lens[num_spk]
      combo = batch_sample[num_spk]['combo'].cuda()

      model.hidden = model.init_hidden(batch)

      sources = []
      for i in range(num_spk):
        source = batch_sample[num_spk]['source'+str(i+1)].cuda()
        source, lens = pad_packed_sequence(source, batch_first=True)
        # source: tensor of shape (batch, seq_length, feat_dim)
        sources.append(source)
      source_usage = [[] for _ in range(num_spk)]
      for dnn_pass in range(num_spk):
        mask_out = model.forward_separate(combo)
        # mask_out: tensor of shape (batch, seq_length, feat_dim)

        combos, lens = pad_packed_sequence(combo, batch_first=True)
        # combos: tensor of shape (batch, seq_length, feat_dim*2)
        mixes = torch.index_select(combos, 2, torch.LongTensor(range(model.feat_dim)).cuda())
        # mixes: tensor of shape (batch, seq_length, feat_dim)
        lengths = lens.float().cuda()

        masked = mask_out * mixes
        losses = torch.stack([torch.sum(sep_loss_fn(masked, source).view(batch, -1), dim=1) for source in sources])

        for source_ind in range(num_spk):
          for index in source_usage[source_ind]:
            losses[source_ind][index] = float("Inf")

        min_losses, indices = torch.min(losses, 0)
        for sample_ind in range(batch):
          source_usage[indices[sample_ind]].append(sample_ind)

        loss += torch.sum(min_losses)/num_spk
        norm += torch.sum(lengths)*model.feat_dim

        if plotdir:
          os.system("mkdir -p "+plotdir)
          if dnn_pass == 0:
            plot.plot_spec(combos[0].detach().cpu().numpy()[:,0:model.feat_dim], plotdir+'/'+str(num_spk)+'-Spk_Mix.png')
          prefix = plotdir+'/'+str(num_spk)+'-Spk_Pass-'+str(dnn_pass+1)+'_'
          plot.plot_spec(combos[0].detach().cpu().numpy(), prefix+'Input.png')
          plot.plot_spec(combos[0].detach().cpu().numpy()[:,model.feat_dim+1:model.feat_dim*2], prefix+'Attenmask.png')
          plot.plot_spec(mask_out[0].detach().cpu().numpy(), prefix+'Mask_Out.png')
          plot.plot_spec(masked[0].detach().cpu().numpy(), prefix+'Masked_Mix.png')
          plot.plot_spec(sources[indices[0]][0].detach().cpu().numpy(), prefix+'Chosen_Source.png')

        spec_zeros = torch.zeros(mask_out.shape).cuda()
        residual_comp = torch.cat((spec_zeros, mask_out), 2)
        combos = F.relu_(combos - residual_comp)
        combo = pack_padded_sequence(combos, lens, batch_first=True)

  return loss/norm, norm

# define test pass
def compute_masks(model, batch_sample, out_dir):
  model.zero_grad()

  for num_spk in range(batch_sample.max_spk):
    if len(batch_sample[num_spk]) > 0:
      combo = batch_sample[num_spk]['combo'].cuda()
      name = batch_sample[num_spk]['name']
      batch = int(combo.batch_sizes[0])

      model.hidden = model.init_hidden(batch)

      dicts = [dict() for _ in range(batch)]
      for dnn_pass in range(num_spk):
        mask_out = model.forward_separate(combo)

        combos, lens = pad_packed_sequence(combo, batch_first=True)
        spec_zeros = torch.zeros(mask_out.shape).cuda()
        residual_comp = torch.cat((spec_zeros, mask_out), 2)
        combos = combos - residual_comp
        combo = pack_padded_sequence(combos, lens, batch_first=True)

        for i in range(len(name)):
          dicts[i]['s'+str(dnn_pass+1)] = mask_out[i].cpu().numpy().transpose()[:,0:lens[i]]

      for i in range(batch):
        np.savez_compressed(out_dir+'/'+name[i], **(dicts[i]))
