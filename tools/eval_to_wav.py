#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
sys.path.append(os.path.abspath("/home/mmaciej2/tools/python/greg_sell/"))
import audio_tools as at


def get_args():
  parser = argparse.ArgumentParser(
    description="""This script reconstructs wav files from a directory of mix
    spectrograms and source masks""")

  parser.add_argument("mix_spec_dir", metavar="mix-spec-dir", type=str,
                      help="Mix spectrogram directory")
  parser.add_argument("mask_dir", metavar="mask-dir", type=str,
                      help="Mask directory")
  parser.add_argument("wav_out_dir", metavar="wav-out-dir", type=str,
                      help="Output wav directory")

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

  wlen = 1.0 * args.fft_dim / args.sample_rate
  overlap = int(args.fft_dim/args.step_size)

  os.system("mkdir -p "+args.wav_out_dir+"/s{1,2}")
  for file in os.listdir(os.path.join(args.mask_dir, 's1')):
    Y = np.load(os.path.join(args.mix_spec_dir, file))
    for source in ['s1', 's2']:
      mask_name = os.path.join(args.mask_dir, source, file)
      wav_name = os.path.join(args.wav_out_dir, source, file.replace('.npy','.wav'))
      mask = np.load(mask_name)
      S = np.multiply(Y, mask)
      s = at.inverse_specgram(S,  winlen=wlen, overlap_add=overlap)
      at.writewav(s, args.sample_rate, wav_name)

if __name__ == '__main__':
  main()
