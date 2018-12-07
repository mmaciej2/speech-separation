#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import mir_eval


def get_args():
  parser = argparse.ArgumentParser(
    description="""This script computes SDR improvement from a directory of
    estimated source wav files to a directory of ground truth wav files. The
    input directory should consist of a 'wav' folder. If there are multiple
    sources per recording, they should be contained in sub-directories of the
    form s1, s2, etc. The script will create a 'sdr' folder at the same level
    of the 'wav' folder and will contain an SDR_results.txt file that contains
    the final SDR statistics as well as a SDR_mix_list.txt file that contains
    the per-source SDRs as well as the file's average SDR.""")

  parser.add_argument("es_dir", metavar="estimated-sources-dir", type=str,
                      help="Estimated source wav directory")
  parser.add_argument("gs_dir", metavar="ground-truth-sources-dir", type=str,
                      help="Ground truth source wav directory")

  parser.add_argument("--num-sources", type=int,
                      help="Number of sources per file",
                      default=2)

  args = parser.parse_args()
  return args

def main():
  args = get_args()

  sdrs = []
  outdir = args.es_dir.replace('/wav','/sdr')
  os.system("mkdir -p "+outdir)

  listF = open(os.path.join(outdir, 'SDR_mix_list.txt'), 'w')

  if args.num_sources == 1:
    base_dir = args.es_dir
  else:
    base_dir = os.path.join(args.es_dir, 's1')
  for file in os.listdir(base_dir):
    for source in range(args.num_sources):
      g, fs = mir_eval.io.load_wav(os.path.join(args.gs_dir, 's'+str(source+1), file))
      e, fs = mir_eval.io.load_wav(os.path.join(args.es_dir, 's'+str(source+1), file))
      if source == 0:
        source_length = len(e)
        gs = np.zeros((args.num_sources, source_length))
        es = np.zeros((args.num_sources, source_length))
      gs[source] = g[0:source_length]
      es[source] = e[0:source_length]

    sdr, sir, par, perm = mir_eval.separation.bss_eval_sources(gs, es)

    list_out = file
    for value in sdr:
      sdrs.append(value)
      lit_out = list_out+' '+str(value)
    list_out = list_out+' '+str(sum(sdr)/args.num_sources)
    listF.write(list_out+'\n')
  listF.close()

  sdrs = np.array(sdrs)
  outF = open(os.path.join(outdir, 'SDR_results.txt'), 'w')
  outF.write("Mean:\t"+str(np.mean(sdrs))+'\n')
  outF.write("Std:\t"+str(np.std(sdrs))+'\n')
  outF.write("Max:\t"+str(np.amax(sdrs))+'\n')
  outF.write("Min:\t"+str(np.amin(sdrs))+'\n')
  outF.close()

if __name__ == '__main__':
  main()
