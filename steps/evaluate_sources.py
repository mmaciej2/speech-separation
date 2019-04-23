#!/usr/bin/env python3

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
  sirs = []
  sars = []
  num_src_dict = load_num_src_dict(args.data_dir+"/utt2num_spk")

  sessionSdrF = open(args.exp_dir+"/results/session_SDRs.txt", 'w')
  sourceSdrF = open(args.exp_dir+"/results/source_SDRs.txt", 'w')
  sessionSirF = open(args.exp_dir+"/results/session_SIRs.txt", 'w')
  sourceSirF = open(args.exp_dir+"/results/source_SIRs.txt", 'w')
  sessionSarF = open(args.exp_dir+"/results/session_SARs.txt", 'w')
  sourceSarF = open(args.exp_dir+"/results/source_SARs.txt", 'w')

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
      sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(oracle_sources, est_sources)

      sessionSdrF.write(ID+' '+str(sum(sdr)/num_src)+'\n')
      sourceSdrF.write(ID)
      for value in sdr:
        sdrs.append(value)
        sourceSdrF.write(' '+str(value))
      sourceSdrF.write('\n')

      sessionSirF.write(ID+' '+str(sum(sir)/num_src)+'\n')
      sourceSirF.write(ID)
      for value in sir:
        sirs.append(value)
        sourceSirF.write(' '+str(value))
      sourceSirF.write('\n')

      sessionSarF.write(ID+' '+str(sum(sar)/num_src)+'\n')
      sourceSarF.write(ID)
      for value in sar:
        sars.append(value)
        sourceSarF.write(' '+str(value))
      sourceSarF.write('\n')

  sessionSdrF.close()
  sourceSdrF.close()
  sessionSirF.close()
  sourceSirF.close()
  sessionSarF.close()
  sourceSarF.close()

  sdrs = np.array(sdrs)
  sirs = np.array(sirs)
  sars = np.array(sars)

  outF = open(args.exp_dir+"/results/SDR_stats.txt", 'w')
  outF.write("Mean:\t"+str(np.mean(sdrs))+'\n')
  outF.write("Std:\t"+str(np.std(sdrs))+'\n')
  outF.write("Max:\t"+str(np.amax(sdrs))+'\n')
  outF.write("Min:\t"+str(np.amin(sdrs))+'\n')
  outF.close()

  outF = open(args.exp_dir+"/results/SIR_stats.txt", 'w')
  outF.write("Mean:\t"+str(np.mean(sirs))+'\n')
  outF.write("Std:\t"+str(np.std(sirs))+'\n')
  outF.write("Max:\t"+str(np.amax(sirs))+'\n')
  outF.write("Min:\t"+str(np.amin(sirs))+'\n')
  outF.close()

  outF = open(args.exp_dir+"/results/SAR_stats.txt", 'w')
  outF.write("Mean:\t"+str(np.mean(sars))+'\n')
  outF.write("Std:\t"+str(np.std(sars))+'\n')
  outF.write("Max:\t"+str(np.amax(sars))+'\n')
  outF.write("Min:\t"+str(np.amin(sars))+'\n')
  outF.close()

if __name__ == '__main__':
  main()
