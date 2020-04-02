#/usr/bin/env python3

import os
import numpy as np
import scipy
import glob
import collections
import re
import librosa
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

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
      return pad_sequence([(torch.from_numpy(b)).float() for b in batch], batch_first=True)
    else:
      return default_collate(batch)

  def __call__(self, batch):
    if not self.key:
      return default_collate(batch)
    else:
      return self.collate_sub_batch(batch)

class TrainSet(Dataset):

  def __init__(self, datadir, location=""):
    self.sr = 8000
    self.length = 4.0

    filelist = datadir+"/wav.scp"

    if location:
      print("tools.copy_scp_data_to_dir.sh "+filelist+" "+location+" --bwlimit 10000 --find-sources true")
      os.system("tools/copy_scp_data_to_dir.sh "+filelist+" "+location+" --bwlimit 10000 --find-sources true")
      self.wav_map = {line.split(' ')[0] : location+'/'+line.rstrip('\n').split(' ')[1] for line in open(filelist)}
    else:
      self.wav_map = {line.split(' ')[0] : line.rstrip('\n').split(' ')[1] for line in open(filelist)}

    if os.path.isfile(datadir+"/segments"):
      self.use_segs = True
      self.list = [line.rstrip('\n').split(' ') for line in open(datadir+"/segments")]
    else:
      self.use_segs = False
      self.list = [line.split(' ')[0] for line in open(filelist)]

    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    if self.use_segs:
      wav_path = self.wav_map[self.list[idx][1]]
      offset = float(self.list[idx][2])
      duration = float(self.list[idx][3])-float(self.list[idx][2])
    else:
      wav_path = self.wav_map[self.list[idx]]
      duration = librosa.core.get_duration(filename=wav_path)
    all_wav_paths = sorted(glob.glob(wav_path.replace("/mix/","/*/")))
    num_src = len(all_wav_paths)-1
    sample = {'num_src': num_src}
    if duration > self.length:
      chunk_offset = np.random.uniform(0.0, duration-self.length)
      duration = self.length
    else:
      chunk_offset = 0.0
    srcs = []
    for i in range(len(all_wav_paths)):
      if self.use_segs:
        audio, fs = librosa.core.load(all_wav_paths[i], sr=self.sr, offset=offset+chunk_offset, duration=duration)
      else:
        audio, fs = librosa.core.load(all_wav_paths[i], sr=self.sr, offset=chunk_offset, duration=duration)
      if i == 0:
        sample['mix'] = audio
      else:
        srcs.append(audio)
    sample['length'] = len(audio)
    sample['srcs'] = np.stack(srcs, axis=1) # [length, n_srcs]
    return sample  # { 'mix', 'srcs', ..., 'num_src', 'length' }

class ValidationSet(Dataset):

  def __init__(self, datadir, location=""):
    self.sr = 8000

    filelist = datadir+"/wav.scp"

    if location:
      print("tools/copy_scp_data_to_dir.sh "+filelist+" "+location+" --bwlimit 10000 --find-sources true")
      os.system("tools/copy_scp_data_to_dir.sh "+filelist+" "+location+" --bwlimit 10000 --find-sources true")
      self.wav_map = {line.split(' ')[0] : location+'/'+line.rstrip('\n').split(' ')[1] for line in open(filelist)}
    else:
      self.wav_map = {line.split(' ')[0] : line.rstrip('\n').split(' ')[1] for line in open(filelist)}

    if os.path.isfile(datadir+"/segments"):
      self.use_segs = True
      self.list = [line.rstrip('\n').split(' ') for line in open(datadir+"/segments")]
    else:
      self.use_segs = False
      self.list = [line.split(' ')[0] for line in open(filelist)]

    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    if self.use_segs:
      wav_path = self.wav_map[self.list[idx][1]]
      offset = float(self.list[idx][2])
      duration = float(self.list[idx][3])-float(self.list[idx][2])
    else:
      wav_path = self.wav_map[self.list[idx]]
    all_wav_paths = sorted(glob.glob(wav_path.replace("/mix/","/*/")))
    num_src = len(all_wav_paths)-1
    sample = {'num_src': num_src}
    srcs = []
    for i in range(len(all_wav_paths)):
      if self.use_segs:
        audio, fs = librosa.core.load(all_wav_paths[i], sr=self.sr, offset=offset, duration=duration)
      else:
        audio, fs = librosa.core.load(all_wav_paths[i], sr=self.sr)
      if i == 0:
        sample['mix'] = audio
      else:
        srcs.append(audio)
    sample['length'] = len(audio)
    sample['srcs'] = np.stack(srcs, axis=1)
    return sample  # { 'mix', 'srcs', ..., 'num_src', 'length' }

class TestSet(Dataset):

  def __init__(self, datadir):
    self.sr = 8000
    filelist = datadir+"/wav.scp"
    self.wav_map = {line.split(' ')[0] : line.rstrip('\n').split(' ')[1] for line in open(filelist)}
    if os.path.isfile(datadir+"/segments"):
      self.use_segs = True
      self.list = [line.rstrip('\n').split(' ') for line in open(datadir+"/segments")]
    else:
      self.use_segs = False
      self.list = [line.split(' ')[0] for line in open(filelist)]
    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    if self.use_segs:
      sample = {'name': self.list[idx][0]}
      wav_path = self.wav_map[self.list[idx][1]]
      offset = float(self.list[idx][2])
      duration = float(self.list[idx][3])-float(self.list[idx][2])
    else:
      sample = {'name': self.list[idx]}
      wav_path = self.wav_map[self.list[idx]]
    all_wav_paths = sorted(glob.glob(wav_path.replace("/mix/","/*/")))
    num_src = len(all_wav_paths)-1
    sample['num_src'] = num_src
    if self.use_segs:
      audio, fs = librosa.core.load(all_wav_paths[0], sr=self.sr, offset=offset, duration=duration)
    else:
      audio, fs = librosa.core.load(all_wav_paths[0], sr=self.sr)
    sample['mix'] = audio
    sample['length'] = len(audio)
    return sample  # { 'mix', 'name', 'num_src', 'length' }


# Define Network

class SkipBLSTM(nn.Module):
  def __init__(self, in_dim, hidden_dim, num_layers=4):
    super(SkipBLSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.layers = nn.ModuleList([nn.LSTM(in_dim, hidden_dim, bidirectional=True)])
    self.layers.extend([nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True) for i in range(num_layers-1)])

  def init_hidden(self, batch_size):
    self.hidden = [(torch.randn(2, batch_size, self.hidden_dim).cuda(),
                    torch.randn(2, batch_size, self.hidden_dim).cuda())
                   for i in range(self.num_layers)]

  def forward(self, x):
    x, self.hidden[0] = self.layers[0](x, self.hidden[0])
    for i in range(1, self.num_layers):
      y, self.hidden[i] = self.layers[i](x, self.hidden[i])
      if i % 2:
        x = y + x
      else:
        x = y
    return x

class SepDNN(nn.Module):
  def set_param(self, kwargs, param, default, conversion_fn=lambda x: x):
    if param in kwargs.keys():
      setattr(self, param, conversion_fn(kwargs[param]))
      print('modelparam:', param, conversion_fn(kwargs[param]))
    else:
      setattr(self, param, default)
      print('modelparam:', param, default)

  def pad_batch(self, x):  # x: [batch, length]
    # This zero pads the batch to be even with window/stride
    length = x.shape[1]
    new_length = np.ceil((length-self.basis_len)/self.stride).astype(int)*self.stride+self.basis_len
    return F.pad(x, (0, new_length-length))

  def __init__(self, gpuid, **kwargs):
    super(SepDNN, self).__init__()

    self.gpuid = gpuid

    self.set_param(kwargs, 'num_bases', 500, int)
    self.set_param(kwargs, 'basis_len', 80, int)
    self.set_param(kwargs, 'stride', 40, int)
    self.set_param(kwargs, 'num_srcs', 2, int)
    self.set_param(kwargs, 'hidden_dim', 600, int)
    self.set_param(kwargs, 'num_layers', 4, int)

    # Network Layers
    self.enc = nn.Conv1d(1, self.num_bases, self.basis_len, stride=self.stride, padding=self.basis_len//2, bias=False)
    self.dec = nn.ConvTranspose1d(self.num_bases, 1, self.basis_len, stride=self.stride, padding=self.basis_len//2, bias=False)

    self.norm = nn.LayerNorm(self.num_bases)
    self.blstm = SkipBLSTM(self.num_bases, self.hidden_dim, num_layers=self.num_layers)
    self.lin = nn.Linear(2*self.hidden_dim, self.num_srcs*self.num_bases)

  def forward(self, x):  # x: [batch, len]
    batch = x.shape[0]
    self.blstm.init_hidden(batch)

    # Extract features
    x = self.pad_batch(x)
    x = torch.unsqueeze(x, 1)  # x: [batch, 1, len]
    feats = F.relu_(self.enc(x)).permute(2, 0, 1)  # feats: [t, batch, n_bases]
    feats = self.norm(feats)

    # Apply BLSTMs
    enc = self.blstm(feats)  # enc: [t, batch, hidden_dim*2]

    # Create masks
    masks = torch.sigmoid(self.lin(enc))  # masks: [t, batch, num_bases*n_srcs]
    masks = torch.reshape(masks, (-1, batch, self.num_srcs, self.num_bases)).permute(1, 2, 3, 0)  # masks: [batch, n_srcs, num_bases, t]

    # Apply masks
    feats = torch.unsqueeze(feats, -1).permute(1, 3, 2, 0)  # feats: [batch, 1, num_bases, t]
    feats = feats * masks  # feats: [batch, n_srcs, num_bases, t]

    # Synthesize audio
    feats = torch.reshape(feats, (batch*self.num_srcs, self.num_bases, -1))
    waveforms = self.dec(feats)  # waveforms: [batch*self.num_srcs, 1, len]
    waveforms = torch.squeeze(waveforms, 1)
    waveforms = torch.reshape(waveforms, (batch,  self.num_srcs, -1))

    return waveforms

def batch_SI_SNR(estimates, sources):  # [batch, srcs, length]
  estimates = estimates - torch.mean(estimates, 2, keepdim=True)
  sources = sources - torch.mean(estimates, 2, keepdim=True)

  power = torch.sum(torch.pow(sources, 2), 2, keepdim=True)
  dot = torch.sum(estimates * sources, 2, keepdim=True)

  sources = dot / power * sources

  noise = estimates - sources

  si_snr = 10 * torch.log10(torch.sum(torch.pow(sources, 2), 2) / torch.sum(torch.pow(noise, 2), 2))  # [batch, srcs]

  return si_snr

def compute_cv_loss(model, epoch, batch_sample, plotdir=""):
  return compute_loss(model, epoch, batch_sample, plotdir=plotdir)

def compute_loss(model, epoch, batch_sample, plotdir=""):
  loss = 0
  norm = 0

  mix = batch_sample['mix'].cuda()
  srcs = batch_sample['srcs'].cuda().permute(0, 2, 1)
  length = batch_sample['length'].cuda()

  batch = len(length)

  model.zero_grad()
  
  est_srcs = model(mix)
  est_srcs = est_srcs[:, :, :srcs.shape[2]]

  for i in range(len(length)):
    est_srcs[i, :, length[i]:] = 0

  losses = torch.stack([-1.0 * torch.mean(batch_SI_SNR(est_srcs[:, torch.LongTensor(perm)], srcs), -1) for perm in itertools.permutations(range(model.num_srcs))], dim=-1)

  min_losses, indices = torch.min(losses, -1)

  loss += torch.sum(min_losses)
  norm += torch.tensor(batch)

  return loss/norm, norm

def estimate_sources(model, batch_sample, out_dir, save_diagnostics=False):
  mix = batch_sample['mix'].cuda()
  name = batch_sample['name']
  length = batch_sample['length'].cuda()

  batch = len(length)

  model.zero_grad()

  est_srcs = model(mix)

  for i in range(len(length)):
    est_srcs[i, :, length[i]:] = 0

  for i in range(batch):
    for src in range(model.num_srcs):
      s = est_srcs[i, src, 0:length[i]].cpu().numpy()
      s = 0.9*s/np.max(np.abs(s))
      wav = s*32767
      scipy.io.wavfile.write(out_dir+"/s"+str(src+1)+'/'+name[i]+".wav", 8000, wav.astype('int16'))
