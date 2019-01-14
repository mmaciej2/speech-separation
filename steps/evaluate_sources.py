#!/usr/bin/env python

import argparse
import numpy as np
import mir_eval


def get_args():
  parser = argparse.ArgumentParser(
    description="""This script computes SDR improvement for a set of estimated
    sources and ground truth sources.""")

  parser.add_argument("data_dir", metavar="data-dir", type=str,
                      help="Test set data directory")
  parser.add_argument("exp_dir", metavar="exp-dir", type=str,
                      help="Experiment directory")

  args = parser.parse_args()
  return args

def load_num_src_dict(utt2num_spk_file):
  num_src = dict()
  with open(utt2num_spk_file, 'r') as num_srcF:
    for line in num_srcF:
      num_src[line.split(' ')[0]] = int(line.rstrip().split(' ')[1])
  return num_src

def main():
  args = get_args()

  sdrs = []
  num_src_dict = load_num_src_dict(args.data_dir+"/utt2num_spk")

  sessionF = open(args.exp_dir+"/results/session_SDRs.txt", 'w')
  sourceF = open(args.exp_dir+"/results/source_SDRs.txt", 'w')

  with open(args.data_dir+"/wav.scp", 'r') as wavF:
    for line in wavF:
      ID = line.rstrip().split(' ')[0]
      oracle_mix_wav = line.rstrip().split(' ')[1]
      num_src = num_src_dict[ID]
      for source in range(num_src):
        oracle_source, fs = mir_eval.io.load_wav(oracle_mix_wav.replace("/mix/", "/s"+str(source+1)+"/"))
        est_source, fs = mir_eval.io.load_wav(args.exp_dir+"/wav/s"+str(source+1)+"/"+ID+".wav")
        if source == 0:
          source_length = len(est_source)
          oracle_sources = np.zeros((num_src, source_length))
          est_sources = np.zeros((num_src, source_length))
        oracle_sources[source] = oracle_source[0:source_length]
        est_sources[source] = est_source[0:source_length]
      sdr, sir, par, perm = mir_eval.separation.bss_eval_sources(oracle_sources, est_sources)

      sessionF.write(ID+' '+str(sum(sdr)/num_src)+'\n')
      sourceF.write(ID)
      for value in sdr:
        sdrs.append(value)
        sourceF.write(' '+str(value))
      sourceF.write('\n')

  sessionF.close()
  sourceF.close()

  sdrs = np.array(sdrs)
  outF = open(args.exp_dir+"/results/SDR_stats.txt", 'w')
  outF.write("Mean:\t"+str(np.mean(sdrs))+'\n')
  outF.write("Std:\t"+str(np.std(sdrs))+'\n')
  outF.write("Max:\t"+str(np.amax(sdrs))+'\n')
  outF.write("Min:\t"+str(np.amin(sdrs))+'\n')
  outF.close()

if __name__ == '__main__':
  main()
