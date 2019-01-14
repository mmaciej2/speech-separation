#!/usr/bin/env python3

import argparse
import numpy as np
import scipy
import librosa


def get_args():
  parser = argparse.ArgumentParser(
    description="""This script reconstructs wav files from mix spectrograms
    and estimated source masks""")

  parser.add_argument("data_dir", metavar="data-dir", type=str,
                      help="Data directory")
  parser.add_argument("exp_dir", metavar="exp-dir", type=str,
                      help="Experiment directory")

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

  with open(args.data_dir+'/feats_test.scp', 'r') as featsF:
    for line in featsF:
      ID = line.rstrip().split(' ')[0]
      mix_spec = np.load(line.rstrip().split(' ')[1])['mix']
      masks = np.load(args.exp_dir+"/masks/"+ID+'.npz')
      for source in masks.files:
        wav_out = args.exp_dir+"/wav/"+source+'/'+ID+".wav"
        S = np.multiply(mix_spec, masks[source])
        s = librosa.core.istft(S, hop_length=args.step_size)
        wav = s*32767
        scipy.io.wavfile.write(wav_out, args.sample_rate, wav.astype('int16'))

if __name__ == '__main__':
  main()
