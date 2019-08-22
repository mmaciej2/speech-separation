#/usr/bin/env python3

import torch
import sys
import os
import glob
import numpy as np
import collections
import re
import librosa
import itertools

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate, numpy_type_map
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
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
      return pad_sequence([(torch.from_numpy(b)).float() for b in batch], batch_first=True)
    else:
      return default_collate(batch)

  def __call__(self, batch):
    if not self.key:
      return default_collate(batch)
    else:
      max_src = max([int(d["num_src"]) for d in batch])
      batch_out = MultiSrcBatch(max_src, len(batch))
      for num_src in range(max_src+1):
        inds = [i for i in range(len(batch)) if int(batch[i]["num_src"]) == num_src]
        batch_out.sub_batch_lens.append(len(inds))
        if len(inds) > 0:
          batch_out.append(self.collate_sub_batch([batch[i] for i in inds]))
        else:
          batch_out.append({})
      return batch_out

class MultiSrcBatch():

  def __init__(self, max_src, length):
    self.sub_batches = list()
    self.sub_batch_lens = list()
    self.max_src = max_src
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
      self.list = [line.rstrip('\n').split(' ') for line in datadir+"/segments"]
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
      duration = float(self.list[idx][3])-float(self.list[idx][3])
    else:
      wav_path = self.wav_map[self.list[idx]]
    all_wav_paths = sorted(glob.glob(wav_path.replace("/mix/","/*/")))
    num_src = len(all_wav_paths)-1
    sample = {'num_src': num_src}
    for i in range(len(all_wav_paths)):
      if self.use_segs:
        audio, fs = librosa.core.load(all_wav_paths[i], sr=self.sr, offset=offset, duration=duration)
      else:
        audio, fs = librosa.core.load(all_wav_paths[i], sr=self.sr)
      if i == 0:
        sample['mix'] = audio
      else:
        sample['s'+str(i)] = audio
    sample['length'] = len(audio)
    return sample  # { 'mix', 's1', 's2', ..., 'num_src', 'length' }

class TestSet(Dataset):

  def __init__(self, datadir):
    self.sr = 8000
    filelist = datadir+"/wav.scp"
    self.wav_map = {line.split(' ')[0] : line.rstrip('\n').split(' ')[1] for line in open(filelist)}
    if os.path.isfile(datadir+"/segments"):
      self.use_segs = True
      self.list = [line.rstrip('\n').split(' ') for line in datadir+"/segments"]
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
      duration = float(self.list[idx][3])-float(self.list[idx][3])
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

# define nnet
class SepDNN(nn.Module):
  def set_param(self, kwargs, param, default, conversion_fn=lambda x: x):
    if param in kwargs.keys():
      setattr(self, param, conversion_fn(kwargs[param]))
      print('modelparam:', param, conversion_fn(kwargs[param]))
    else:
      setattr(self, param, default)
      print('modelparam:', param, default)

  def __init__(self, gpuid, **kwargs):
    super(SepDNN, self).__init__()

    self.gpuid = gpuid

    self.set_param(kwargs, 'num_basis_vec', 257, int)
    self.set_param(kwargs, 'basis_len', 512, int)
    self.set_param(kwargs, 'stride', 128, int)
    self.set_param(kwargs, 'num_src', 2, int)

    # Feature extraction layers

    self.conv_extract = nn.Conv1d(1, self.num_basis_vec, self.basis_len, stride=self.stride)

    # Reconstruction layers

    self.tconv_synth = nn.ConvTranspose1d(self.num_basis_vec, 1, self.basis_len, stride=self.stride)

    # Separation layers

    self.blstm = nn.LSTM(self.num_basis_vec, 600, num_layers=2, bidirectional=True)
    self.bn = nn.BatchNorm1d(600*2)
    self.lin = nn.Linear(600*2, self.num_basis_vec*self.num_src)

  def init_hidden(self, batch_size):
    return (torch.randn(2*2, batch_size, 600).cuda(),
            torch.randn(2*2, batch_size, 600).cuda())

  def pad_batch(self, batch):
    # This zero pads the batch to be even with window/stride
    length = batch.shape[1]
    new_length = np.ceil((length-self.basis_len)/self.stride).astype(int)*self.stride+self.basis_len
    return F.pad(batch, (0, new_length-length))

  def extract(self, x):
    x = self.pad_batch(x)
    # x: tensor of shape (batch, length-time)

    x = x.view((x.shape[0], 1, x.shape[1]))
    # x: tensor of shape (batch, 1, length-time)

    x = F.relu_(self.conv_extract(x))
    # x: tensor of shape (batch, num_basis_vec, length-feat)

    return x

#  def synthesize(self, x, length):
  def synthesize(self, x):
    # x: tensor of shape (batch, num_basis_vec, length-feat)

    # mask out padding convolutions
#    feat_lengths = (length-self.basis_len)/self.stride+1
#    mask = torch.zeros(x.shape)
#    for i, feat_len in enumerate(feat_lengths):
#      mask[i, :, 0:feat_len] = torch.ones((1, mask.shape[1], feat_len))
#    x = x * mask

    x = self.tconv_synth(x)
    # x: tensor of shape (batch, 1, length-time)

    x = x.view((x.shape[0], x.shape[2]))
    # x: tensor of shape (batch, length-time)

    return x

  def separate(self, x, lengths):
    # x: tensor of shape (batch, num_basis_vec, feat_length)

    x = pack_padded_sequence(x.permute(0,2,1), lengths, batch_first=True)
    # x: packed sequence of dim  num_basis_vec

    x, self.hidden = self.blstm(x, self.hidden)
    # x: packed sequence of dim 600*2

    x, lens = pad_packed_sequence(x, batch_first=True)
    # x: tensor of shape (batch, feat_length, 600*2)

    x = self.bn(x.permute(0,2,1).contiguous()).permute(0,2,1)
    # x: tensor of shape (batch, feat_length, 600*2)

    x = self.lin(x)
    # x: tensor of shape (batch, feat_length, num_basis_vec*num_src)

    x = torch.sigmoid(x)
    # x: tensor of shape (batch, feat_length, num_basis_vec*num_src)

    return x

  def forward(self, x, lengths):
    # x: tensor of shape (batch, length-time)

    x = self.extract(x)
    # x: tensor of shape (batch, num_basis_vec, length-feat)

    feat_lengths = torch.ceil((lengths.type(torch.float)-self.basis_len)/self.stride+1)
    x = self.separate(x, feat_lengths)
    # x: tensor of shape (batch, length-feat, num_basis_vec*num_src)

    est_sources = []
    for i in range(self.num_src):
      est_sources.append(self.synthesize(torch.index_select(x, 2, torch.LongTensor(range(i*self.num_basis_vec, (i+1)*self.num_basis_vec)).cuda()).permute(0,2,1).contiguous()))
    x = torch.stack(est_sources, dim=1)
    # x: (batch, num_src, length-time)

    return x

def SI_SNR(est_sources, oracle_sources):
  # sources: (batch, num_src, length-time)

  scaler = torch.squeeze(torch.matmul(torch.unsqueeze(est_sources,2), torch.unsqueeze(oracle_sources,3))) / torch.squeeze(torch.matmul(torch.unsqueeze(oracle_sources,2), torch.unsqueeze(oracle_sources,3)))
  # scaler: (batch, num_src)

  s_target = torch.matmul(torch.diag_embed(scaler), oracle_sources)
  e_noise = est_sources-s_target
  # s,e: (batch, num_src, length-time)

  magsq_s_target = torch.squeeze(torch.matmul(torch.unsqueeze(s_target,2), torch.unsqueeze(s_target,3)))
  magsq_e_noise = torch.squeeze(torch.matmul(torch.unsqueeze(e_noise,2), torch.unsqueeze(e_noise,3)))
  # magsq: (batch, num_src)

  si_snr = 10*(torch.log10(magsq_s_target) - torch.log10(magsq_e_noise))
  # si_snr: (batch, num_src)

  return torch.sum(si_snr, 1)  # (batch)

def compute_cv_loss(model, epoch, batch_sample, plotdir=""):
  if plotdir:
    loss, norm = compute_loss(model, epoch, batch_sample, plotdir)
  else:
    loss, norm = compute_loss(model, epoch, batch_sample)
  return loss, norm

# define training pass
def compute_loss(model, epoch, batch_sample, plotdir=""):
  loss = 0
  norm = 0

  for num_src in range(batch_sample.max_src+1):
    if batch_sample.sub_batch_lens[num_src] > 0:
      sample_size = batch_sample.sub_batch_lens[num_src]
      sample = batch_sample[num_src]  # { 'mix', 's1', 's2', ..., 'num_src', 'length' }

      mix = sample['mix'].cuda()
      length = sample['length'].cuda()
      sources = []
      for i in range(model.num_src):
        sources.append(model.pad_batch(sample['s'+str(i+1)].cuda()))

      model.zero_grad()
      model.hidden = model.init_hidden(sample_size)

      est_sources = model(mix, length)
      # est_sources: tensor of shape (batch, num_src, length-time)

      perms = list(itertools.permutations(range(model.num_src)))
      losses = torch.stack([SI_SNR(est_sources, oracle_sources_permute) for oracle_sources_permute in [torch.cat([torch.unsqueeze(sources[i], 1) for i in perm], dim=1) for perm in perms]])

      min_losses, indices = torch.min(losses, 0)

      loss -= torch.sum(min_losses)/num_src  # loss per speaker
      norm += torch.tensor(sample_size)  # loss per mixture

      if plotdir:
        os.system("mkdir -p "+plotdir)
        mix_mag_spec = np.abs(librosa.core.stft(mix[0].detach().cpu().numpy(), n_fft=model.basis_len, hop_length=model.stride))
        plot.plot_spec(mix_mag_spec.T, plotdir+'/STFT-Mixture.png')
        basis_vecs = torch.squeeze(model.conv_extract.weight).detach().cpu().numpy()
        mag_fft_basis = np.abs(np.fft.fft(basis_vecs))
        plot.plot_spec(mag_fft_basis[:,:int(mag_fft_basis.shape[1]/2)-1:-1], plotdir+'/basis_vec_spectra.png')
        for i in range(num_src):
          est_mag_spec = np.abs(librosa.core.stft(est_sources[0,i,:].detach().cpu().numpy(), n_fft=model.basis_len, hop_length=model.stride))
          plot.plot_spec(est_mag_spec.T, plotdir+'/STFT-estimated-'+str(i+1)+'.png')
          oracle_mag_spec = np.abs(librosa.core.stft(sources[perms[indices[0]][i]][0].detach().cpu().numpy(), n_fft=model.basis_len, hop_length=model.stride))
          plot.plot_spec(oracle_mag_spec.T, plotdir+'/STFT-oracle-'+str(i+1)+'.png')

  return loss/norm, norm

# define test pass
def estimate_sources(model, batch_sample, out_dir):
  model.zero_grad()

  for num_src in range(batch_sample.max_src):
    if batch_sample.sub_batch_lens[num_src] > 0:
      sample_size = batch_sample.sub_batch_lens[num_src]
      mix = batch_sample[num_src]['mix'].cuda()
      name = batch_sample[num_src]['name']
      length = batch_sample[num_src]['length']

      model.hidden = model.init_hidden(batch)

      est_sources = model(mix, length.cuda())
      # est_sources: tensor of shape (batch, num_src, length-time)

      for i in range(batch):
        for src in num_src:
          s = est_sources[i, src, 0:length[i]].cpu().numpy()
          wav = s*32767
          scipy.io.wavfile.write(out_dir+"/s"+str(src+1)+'/'+name[i]+".wav", 8000, wav.astype('int16'))
