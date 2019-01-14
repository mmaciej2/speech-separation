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
  parser.add_argument("data_type", metavar="data-type", type=str,
                      choices=['train', 'test'],
                      help="""Dataset type. Train stores magnitude spectra for
                      mixture and all sources. Test stores just mix spectrum.""")
  parser.add_argument("feat_dir", metavar="feat-dir", type=str,
                      help="Output directory for features")

  parser.add_argument("--fft-dim", type=int,
                      help="Dimension of FFT",
                      default=512)
  parser.add_argument("--step-size", type=int,
                      help="STFT step size",
                      default=128)
  parser.add_argument("--sample-rate", type=int,
                      help="Audio sample rate",
                      default=8000)

  args = parser.parse_args()
  return args

def main():
  args = get_args()

  os.system("mkdir -p "+args.feat_dir)
  featF = open(args.data_dir+"/feats_"+args.data_type+".scp", 'w')
  utt2num_spkF = open(args.data_dir+"/utt2num_spk", 'w')

  with open(args.data_dir+"/wav.scp", 'r') as listF:
    for line in listF:
      file_dict = dict()
      reco_id, filename = line.rstrip().split(' ')
      wav_files = sorted(glob.glob(filename.replace("/mix/","/*/")))
      num_spk = len(wav_files)-1
      if args.data_type == "train":
        for i in range(len(wav_files)):
          audio, fs = librosa.core.load(wav_files[i], sr=args.sample_rate)
          if i == 0:
            file_dict['mix'] = np.abs(librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size))
          else:
            file_dict['s'+str(i)] = np.abs(librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size))
      elif args.data_type == "test":
        audio, fs = librosa.core.load(filename, sr=args.sample_rate)
        file_dict['mix'] = librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size)
      np.savez_compressed(os.path.join(args.feat_dir, reco_id), **file_dict)
      featF.write(reco_id+' '+os.path.join(args.feat_dir, reco_id)+'.npz\n')
      utt2num_spkF.write(reco_id+' '+str(num_spk)+'\n')

  featF.close()
  utt2num_spkF.close()

if __name__ == '__main__':
  main()
