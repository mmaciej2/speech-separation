#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np
import librosa

def get_args():
  parser = argparse.ArgumentParser(
    description="""Extracts and saves STFT-based features for source
    separation""")

  parser.add_argument("data_dir", metavar="data-dir", type=str,
                      help="Data directory with wav.scp")
  parser.add_argument("feat_dir", metavar="feat-dir", type=str,
                      help="Output directory for features")

  parser.add_argument("--batch-window", type=int,
                      help="Batch window length",
                      default=100)
  parser.add_argument("--batch-shift", type=int,
                      help="Batch window shift",
                      default=50)
  parser.add_argument("--dB-thresh", type=float,
                      default=-40)
  parser.add_argument("--fft-dim", type=int,
                      help="Dimension of FFT",
                      default=256)
  parser.add_argument("--step-size", type=int,
                      help="STFT step size",
                      default=64)
  parser.add_argument("--sample-rate", type=int,
                      help="Audio sample rate",
                      default=8000)

  args = parser.parse_args()
  return args

def main():
  args = get_args()

  os.system("mkdir -p "+args.feat_dir)
  featF = open(args.data_dir+"/feats_dpcl_train.scp", 'w')
  utt2num_spkF = open(args.data_dir+"/utt2num_spk", 'w')

  with open(args.data_dir+"/wav.scp", 'r') as listF:
    for line in listF:
      file_dict = dict()
      reco_id, filename = line.rstrip().split(' ')
      wav_files = sorted(glob.glob(filename.replace("/mix/","/*/")))
      num_spk = len(wav_files)-1

      source_mags = list()
      for i in range(len(wav_files)):
        audio, fs = librosa.core.load(wav_files[i], sr=args.sample_rate)
        if i == 0:
          mix = np.abs(librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size)).T
          mix_max = np.amax(mix)
          min_mag = 10**(args.dB_thresh/20) * mix_max
          min_mask = mix > min_mag
        else:
          source_mags.append(np.abs(librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size)).T)

      masks = list()
      for i in range(len(source_mags)):
        mask = min_mask
        for j in [j for j in range(len(source_mags)) if j != i]:
          mask = mask * (source_mags[i] > source_mags[j])
        masks.append(mask.astype(float))

      mixes = list()
      affinities = list()
      affinity_dim = (int(args.fft_dim/2)+1)*args.batch_window
      for i in range(args.batch_window, len(mix), args.batch_shift):
        mixes.append(mix[i-args.batch_window:i])
        affinity = np.stack([mask[i-args.batch_window:i] for mask in masks], axis=2)
        affinities.append(affinity)
      if (len(mix)-args.batch_window)%args.batch_shift != 0:
        mixes.append(mix[-args.batch_window:])
        affinity = np.stack([mask[-args.batch_window:] for mask in masks], axis=2)
        affinities.append(affinity)

      file_dict['mix'] = mixes
      file_dict['affinity'] = affinities
      np.savez_compressed(os.path.join(args.feat_dir, reco_id), **file_dict)
      featF.write(reco_id+' '+os.path.join(args.feat_dir, reco_id)+'.npz\n')
      utt2num_spkF.write(reco_id+' '+str(num_spk)+'\n')

  featF.close()
  utt2num_spkF.close()

if __name__ == '__main__':
  main()
