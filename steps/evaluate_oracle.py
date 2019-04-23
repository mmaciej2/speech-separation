#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np
import librosa
import mir_eval

def get_args():
  parser = argparse.ArgumentParser(
    description="""Computes oracle SDR, SIR, and SAR for a source separation
    data directory""")

  parser.add_argument("data_dir", metavar="data-dir", type=str,
                      help="Data directory")

  parser.add_argument("--hard-mask", action='store_true',
                      help="Use hard mask",
                      default=False)
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

def write_results(ID, sdr, sir, sar, sessionFs, sourceFs, num_src):
  sessionFs[0].write(ID+' '+str(sum(sdr)/num_src)+'\n')
  sessionFs[1].write(ID+' '+str(sum(sir)/num_src)+'\n')
  sessionFs[2].write(ID+' '+str(sum(sar)/num_src)+'\n')
  for F in sourceFs:
    F.write(ID)
  for i in range(num_src):
    sourceFs[0].write(' '+str(sdr[i]))
    sourceFs[1].write(' '+str(sir[i]))
    sourceFs[2].write(' '+str(sar[i]))
  for F in sourceFs:
    F.write('\n')

def main():
  args = get_args()

  if "SGE_TASK_ID" in os.environ.keys():
    if os.environ["SGE_TASK_ID"] == 'undefined':
      job_suffix = ''
    else:
      job_suffix = '.'+os.environ["SGE_TASK_ID"]
  else:
    job_suffix = ''

  if args.hard_mask:
    dir_out = args.data_dir+"/oracle_hard_mask_eval/"
  else:
    dir_out = args.data_dir+"/oracle_soft_mask_eval/"
  os.system("mkdir -p "+dir_out)

  if os.path.isfile(args.data_dir+"/segments"+job_suffix):
    use_seg = True
    seg_dict = {}
    for line in open(args.data_dir+"/segments"+job_suffix):
      seg = line.rstrip().split()
      if seg[1] not in seg_dict.keys():
        seg_dict[seg[1]] = []
      seg_dict[seg[1]].append((seg[0], float(seg[2]), float(seg[3])))
  else:
    use_segs = False

  SDRs = []
  SIRs = []
  SARs = []

  sessionFs = [
    open(dir_out+"/session_SDRs.txt"+job_suffix, 'w'),
    open(dir_out+"/session_SIRs.txt"+job_suffix, 'w'),
    open(dir_out+"/session_SARs.txt"+job_suffix, 'w')]
  sourceFs = [
    open(dir_out+"/source_SDRs.txt"+job_suffix, 'w'),
    open(dir_out+"/source_SIRs.txt"+job_suffix, 'w'),
    open(dir_out+"/source_SARs.txt"+job_suffix, 'w')]

  with open(args.data_dir+"/wav.scp"+job_suffix, 'r') as listF:
    for line in listF:
      reco_id, filename = line.rstrip().split(' ')
      wav_files = sorted(glob.glob(filename.replace("/mix/","/*/")))
      num_src = len(wav_files)-1
      if use_segs:
        for seg in seg_dict[reco_id]:
          for i in rage(len(wav_files)):
            audio, fs = librosa.core.load(wav_files[i], sr=args.sample_rate, offset=seg[1], duration=seg[2]-seg[1])
            if i == 0:
              source_length = len(audio)
              oracle_sources = np.zeros((num_src, source_length))
              est_sources = np.zeros((num_src, source_length))
              mix_spec = librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size)
              oracle_mags = np.zeros((num_src, mix_spec.shape[0], mix_spec.shape[1]))
              oracle_masks = np.zeros(oracle_mags.shape)
            else:
              oracle_sources[i-1] = audio
              oracle_mags[i-1] = np.abs(librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size))
          if args.hard_mask:
            spec_coef_ID = np.argmax(oracle_mags, axis=0)
            for i in range(num_src):
              oracle_masks[i] = (spec_coef_ID == i).astype(float)
          else:
            for i in range(num_src):
              oracle_masks[i] = np.divide(oracle_mags[i], np.abs(mix_spec))
          for i in range(num_src):
            S = np.multiply(mix_spec, oracle_mask[i])
            s = librosa.core.istft(S, hop_length=args.step_size)
            est_sources[i][0:len(s)] = s
          sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(oracle_sources, est_sources, compute_permutation=False)
          write_results(seg, sdr, sir, sar, sessionFs, sourceFs, num_src)
      else:
        for i in range(len(wav_files)):
          audio, fs = librosa.core.load(wav_files[i], sr=args.sample_rate)
          if i == 0:
            source_length = len(audio)
            oracle_sources = np.zeros((num_src, source_length))
            est_sources = np.zeros((num_src, source_length))
            mix_spec = librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size)
            oracle_mags = np.zeros((num_src, mix_spec.shape[0], mix_spec.shape[1]))
            oracle_masks = np.zeros(oracle_mags.shape)
          else:
            oracle_sources[i-1] = audio
            oracle_mags[i-1] = np.abs(librosa.core.stft(audio, n_fft=args.fft_dim, hop_length=args.step_size))
        if args.hard_mask:
          spec_coef_ID = np.argmax(oracle_mags, axis=0)
          for i in range(num_src):
            oracle_masks[i] = (spec_coef_ID == i).astype(float)
        else:
          for i in range(num_src):
            oracle_masks[i] = np.divide(oracle_mags[i], np.abs(mix_spec))
        for i in range(num_src):
          S = np.multiply(mix_spec, oracle_masks[i])
          s = librosa.core.istft(S, hop_length=args.step_size)
          est_sources[i][0:len(s)] = s
        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(oracle_sources, est_sources, compute_permutation=False)
        write_results(reco_id, sdr, sir, sar, sessionFs, sourceFs, num_src)

  for F in sessionFs:
    F.close()
  for F in sourceFs:
    F.close()

if __name__ == '__main__':
  main()
