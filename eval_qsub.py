#!/usr/bin/env python3

import torch
import argparse
import sys
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('models')

def get_args():
  parser = argparse.ArgumentParser(
    description="""This generates and saves output for a test set""")

  parser.add_argument("arch_file", metavar="arch-file", type=str,
                      help="DNN architecture file")
  parser.add_argument("gpu_id", metavar="gpu-id", type=int,
                      help="GPU ID")
  parser.add_argument("model", type=str,
                      help="Trained model to use")
  parser.add_argument("filelist", type=str,
                      help="Test set filelist")
  parser.add_argument("dirout", type=str,
                      help="Output directory")

  parser.add_argument("--batch-size", type=int,
                      help="Batch size",
                      default=100)

  args = parser.parse_args()
  return args

def main():
  args = get_args()
  global m
  print("Using "+args.arch_file+" DNN architecture")
  m = __import__(args.arch_file)

  print("Using GPU", args.gpu_id)
  torch.cuda.set_device(args.gpu_id)
  tmp = torch.ByteTensor([0])
  tmp.cuda()

  print("loading dataset")
  dataset = m.TestSet2mix(args.filelist)
  if args.batch_size <= len(dataset):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collator)
  else:
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=dataset.collator)

  print("loading model")
  model = m.EnhBLSTM(args.gpu_id)
  model.cuda()
  model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage.cuda()))

  with torch.no_grad():
    for i_batch, sample_batch in enumerate(dataloader):
      m.compute_masks(model, sample_batch, args.dirout)

if __name__ == '__main__':
  main()
