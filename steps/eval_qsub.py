#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np

sys.path.append('tools')
import plot

import torch
from torch.utils.data import DataLoader

def get_args():
  parser = argparse.ArgumentParser(
    description="""This generates and saves output for a test set""")

  parser.add_argument("arch_file", metavar="arch-file", type=str,
                      help="DNN architecture file")
  parser.add_argument("gpu_id", metavar="gpu-id", type=int,
                      help="GPU ID")
  parser.add_argument("model", type=str,
                      help="Trained model to use")
  parser.add_argument("data_dir", metavar="data-dir", type=str,
                      help="Test data directory")
  parser.add_argument("dirout", type=str,
                      help="Output directory")

  parser.add_argument("--model-config", type=str,
                      help="Config file for DNN",
                      default="")
  parser.add_argument("--batch-size", type=int,
                      help="Batch size",
                      default=100)

  args = parser.parse_args()
  return args

def main():
  args = get_args()
  global m
  print("Using "+args.arch_file+" DNN architecture")
  sys.path.append(os.path.dirname(args.arch_file))
  m = __import__(os.path.splitext(os.path.basename(args.arch_file))[0])

  print("Using GPU", args.gpu_id)
  torch.cuda.set_device(0)
  tmp = torch.ByteTensor([0])
  tmp.cuda()

  print("loading dataset")
  dataset = m.TestSet(args.data_dir)
  if args.batch_size <= len(dataset):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collator)
  else:
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=dataset.collator)

  print("loading model")
  if args.model_config:
    kwargs = dict()
    for line in open(args.model_config):
      kwargs[line.split('=')[0]] = line.rstrip().split('=')[1]
    model = m.SepDNN(args.gpu_id, **kwargs)
  else:
    model = m.SepDNN(args.gpu_id)
  model.cuda()
  model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage.cuda()))

  model.eval()
  with torch.no_grad():
    for i_batch, sample_batch in enumerate(dataloader):
      m.compute_masks(model, sample_batch, args.dirout)

if __name__ == '__main__':
  main()
