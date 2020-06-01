import os
import os.path
import numpy as np
import glob
import collections
import re
import soundfile as sf

import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence


def load_conf(path):
  kwargs = dict()
  if os.path.isfile(path):
    for line in open(path, 'r'):
      kwargs[line.split('=')[0]] = line.rstrip().split('=')[1]
  return kwargs

def set_param(obj, kwargs, param, default, conversion_fn=lambda x: x):
  if param in kwargs.keys():
    setattr(obj, param, conversion_fn(kwargs[param]))
    print('dataset param:', param, conversion_fn(kwargs[param]))
  else:
    setattr(obj, param, default)
    print('dataset param:', param, default)

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
    conf_args = load_conf(datadir+'/conf.txt')
    set_param(self, conf_args, 'sample_rate', 8000, int)
    set_param(self, conf_args, 'length', 4.0, float)

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
      duration = sf.info(wav_path).duration
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
        audio, fs = sf.read(all_wav_paths[i], start=int((offset+chunk_offset)*self.sample_rate), frames=int(duration*self.sample_rate), dtype=np.float32)
      else:
        audio, fs = sf.read(all_wav_paths[i], start=int(chunk_offset*self.sample_rate), frames=int(duration*self.sample_rate), dtype=np.float32)
      if i == 0:
        sample['mix'] = audio
      else:
        srcs.append(audio)
    assert(fs == self.sample_rate)
    sample['length'] = len(audio)
    sample['srcs'] = np.stack(srcs, axis=1) # [length, n_srcs]
    return sample  # { 'mix', 'srcs', ..., 'num_src', 'length' }

class ValidationSet(Dataset):

  def __init__(self, datadir, location=""):
    conf_args = load_conf(datadir+'/conf.txt')
    set_param(self, conf_wargs, 'sample_rate', 8000, int)

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
        audio, fs = sf.read(all_wav_paths[i], start=int(offset*self.sample_rate), frames=int(duration*self.sample_rate), dtype=np.float32)
      else:
        audio, fs = sf.read(all_wav_paths[i], dtype=np.float32)
      if i == 0:
        sample['mix'] = audio
      else:
        srcs.append(audio)
    assert(fs == self.sample_rate)
    sample['length'] = len(audio)
    sample['srcs'] = np.stack(srcs, axis=1)
    return sample  # { 'mix', 'srcs', ..., 'num_src', 'length' }

class TestSet(Dataset):

  def __init__(self, datadir):
    conf_args = load_conf(datadir+'/conf.txt')
    set_param(self, conf_wargs, 'sample_rate', 8000, int)

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
      audio, fs = sf.read(all_wav_paths[0], start=int(offset*self.sample_rate), frames=int(duration*self.sample_rate), dtype=np.float32)
    else:
      audio, fs = sf.read(all_wav_paths[0], dtype=np.float32)
    assert(fs == self.sample_rate)
    sample['mix'] = audio
    sample['length'] = len(audio)
    return sample  # { 'mix', 'name', 'num_src', 'length' }

class TrainSet_NoisyOracle(Dataset):

  def __init__(self, datadir, location=""):
    conf_args = load_conf(datadir+'/conf.txt')
    set_param(self, conf_args, 'sample_rate', 8000, int)
    set_param(self, conf_args, 'length', 4.0, float)
    set_param(self, conf_args, 'snr', 10, float)
    set_param(self, conf_args, 'target', 'noisy')

    assert(self.target in ['noisy', 'clean'])

    self.rng = np.random.default_rng()

    filelist = datadir+"/wav.scp"

    self.list = [line.split(' ')[0] for line in open(filelist)]

    self.wav_map = {line.split(' ')[0] : line.rstrip('\n').split(' ')[1] for line in open(filelist)}
    self.e_map = np.load(datadir+'/e_map.npz')

    self.noise_map = dict()
    with open(datadir+'/noise_map', 'r') as noiseF:
        for line in noiseF:
            split = line.split()
            self.noise_map[split[0]] = split[1]

    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_id = self.list[idx]
    wav_path = self.wav_map[mix_id]

    mix_info = sf.info(wav_path)
    mix_dur = mix_info.duration
    mix_len = mix_info.frames

    sample = {'num_src': 2}
    if mix_dur > self.length:
      offset = self.rng.integers(0, int((mix_dur-self.length)*self.sample_rate))
    else:
      offset = 0

    s1_audio, fs = sf.read(wav_path.replace('mix_both', 's1_anechoic'), start=offset, frames=int(self.length*self.sample_rate), dtype=np.float32)
    s2_audio, fs = sf.read(wav_path.replace('mix_both', 's2_anechoic'), start=offset, frames=int(self.length*self.sample_rate), dtype=np.float32)
    assert(fs == self.sample_rate)

    noise_path = wav_path.replace('min', 'max').replace('mix_both', 'noise')

    s1_noise, fs = sf.read(noise_path, start=offset, frames=len(s1_audio), dtype=np.float32)
    s2_noise, fs = sf.read(noise_path.replace(mix_id, self.noise_map[mix_id]), start=-mix_len+offset, frames=len(s1_audio), dtype=np.float32)

    rms_amps = self.e_map[mix_id]
    s1_noise *= rms_amps[0]/rms_amps[2]*10**(-self.snr/20)
    s2_noise *= rms_amps[1]/rms_amps[3]*10**(-self.snr/20)

    sample['mix'] = s1_audio + s2_audio + s1_noise + s2_noise

    if self.target == 'noisy':
      sample['srcs'] = np.stack([s1_audio+s1_noise, s2_audio+s2_noise], axis=1)
    else:
      sample['srcs'] = np.stack([s1_audio, s2_audio], axis=1)

    sample['length'] = len(s1_audio)

    return sample  # { 'mix', 'srcs', ..., 'num_src', 'length' }

class ValidationSet_NoisyOracle(Dataset):

  def __init__(self, datadir, location=""):
    conf_args = load_conf(datadir+'/conf.txt')
    set_param(self, conf_args, 'sample_rate', 8000, int)
    set_param(self, conf_args, 'snr', 10, float)
    set_param(self, conf_args, 'target', 'noisy')

    assert(self.target in ['noisy', 'clean'])

    filelist = datadir+"/wav.scp"

    self.list = [line.split(' ')[0] for line in open(filelist)]

    self.wav_map = {line.split(' ')[0] : line.rstrip('\n').split(' ')[1] for line in open(filelist)}
    self.e_map = np.load(datadir+'/e_map.npz')

    self.noise_map = dict()
    with open(datadir+'/noise_map', 'r') as noiseF:
        for line in noiseF:
            split = line.split()
            self.noise_map[split[0]] = split[1]

    self.collator = Collator('mix')

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_id = self.list[idx]
    wav_path = self.wav_map[mix_id]

    mix_len = sf.info(wav_path).frames

    sample = {'num_src': 2}

    s1_audio, fs = sf.read(wav_path.replace('mix_both', 's1_anechoic'), dtype=np.float32)
    s2_audio, fs = sf.read(wav_path.replace('mix_both', 's2_anechoic'), dtype=np.float32)
    assert(fs == self.sample_rate)

    noise_path = wav_path.replace('min', 'max').replace('mix_both', 'noise')

    s1_noise, fs = sf.read(noise_path, frames=mix_len, dtype=np.float32)
    s2_noise, fs = sf.read(noise_path.replace(mix_id, self.noise_map[mix_id]), start=-mix_len, dtype=np.float32)

    rms_amps = self.e_map[mix_id]
    s1_noise *= rms_amps[0]/rms_amps[2]*10**(-self.snr/20)
    s2_noise *= rms_amps[1]/rms_amps[3]*10**(-self.snr/20)

    sample['mix'] = s1_audio + s2_audio + s1_noise + s2_noise

    if self.target == 'noisy':
      sample['srcs'] = np.stack([s1_audio+s1_noise, s2_audio+s2_noise], axis=1)
    else:
      sample['srcs'] = np.stack([s1_audio, s2_audio], axis=1)

    sample['length'] = len(s1_audio)

    return sample  # { 'mix', 'srcs', ..., 'num_src', 'length' }

class TestSet_NoisyOracle(Dataset):

  def __init__(self, datadir):
    conf_args = load_conf(datadir+'/conf.txt')
    set_param(self, conf_wargs, 'sample_rate', 8000, int)

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
#    if self.use_segs:
#      audio, fs = librosa.core.load(all_wav_paths[0], sr=self.sample_rate, offset=offset, duration=duration)
#    else:
#      audio, fs = librosa.core.load(all_wav_paths[0], sr=self.sample_rate)
#    sample['mix'] = audio
#    sample['length'] = len(audio)
    return sample  # { 'mix', 'name', 'num_src', 'length' }

