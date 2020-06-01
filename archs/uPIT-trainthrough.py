#/usr/bin/env python3

import numpy as np
import scipy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from components import conv_stft
from datasets import E2E

TrainSet = E2E.TrainSet
ValidationSet = E2E.ValidationSet
TestSet = E2E.TestSet

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
    new_length = np.ceil((length-self.window)/self.shift).astype(int)*self.shift+self.window
    return F.pad(x, (0, new_length-length))

  def __init__(self, gpuid, **kwargs):
    super(SepDNN, self).__init__()

    self.gpuid = gpuid

    self.set_param(kwargs, 'window', 256, int)
    self.set_param(kwargs, 'shift', 64, int)
    self.set_param(kwargs, 'num_srcs', 2, int)
    self.set_param(kwargs, 'hidden_dim', 600, int)
    self.set_param(kwargs, 'num_layers', 4, int)

    # Network Layers
    self.STFT = conv_stft.ConvSTFT(self.window, self.shift)

    self.norm = nn.LayerNorm(self.window//2+1)
    self.blstm = SkipBLSTM(self.window//2+1, self.hidden_dim, num_layers=self.num_layers)
    self.lin = nn.Linear(2*self.hidden_dim, self.num_srcs*(self.window//2+1))

  def forward(self, x):  # x: [batch, len]
    batch = x.shape[0]
    self.blstm.init_hidden(batch)

    # Extract features
    x = self.pad_batch(x)
    x = torch.unsqueeze(x, 1)  # x: [batch, 1, len]
#    feats = F.relu_(self.enc(x)).permute(2, 0, 1)  # feats: [t, batch, n_bases]
    stft = self.STFT.encode(x)  # stft: [batch, f, t]
    feats = conv_stft.stft_mag(stft).permute(2, 0, 1)  # feats: [t, batch, f//2+1]
    feats = self.norm(feats)

    # Apply BLSTMs
    enc = self.blstm(feats)  # enc: [t, batch, hidden_dim*2]

    # Create masks
    masks = torch.sigmoid(self.lin(enc))  # masks: [t, batch, (f//2+1)*n_srcs]
    masks = torch.reshape(masks, (-1, batch, self.num_srcs, self.window//2+1)).permute(1, 2, 3, 0)  # masks: [batch, n_srcs, f//2+1, t]

    # Apply masks
#    feats = torch.unsqueeze(feats, -1).permute(1, 3, 2, 0)  # feats: [batch, 1, num_bases, t]
#    feats = feats * masks  # feats: [batch, n_srcs, num_bases, t]
    feats = stft.unsqueeze(-1).permute(0, 3, 1, 2)  # [batch, 1, f, t]
    feats = conv_stft.apply_mask(feats, masks)

    # Synthesize audio
    feats = torch.reshape(feats, (batch*self.num_srcs, self.window, -1))
#    waveforms = self.dec(feats)  # waveforms: [batch*self.num_srcs, 1, len]
    waveforms = self.STFT.decode(feats)  # waveforms: [batch*self.num_srcs, 1, len]
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
