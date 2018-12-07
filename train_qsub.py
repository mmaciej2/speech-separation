#!/usr/bin/env python3

import torch
import argparse
import sys
import os
import numpy as np
import collections
import re

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('models')

import quick_plot as qp

def get_args():
  parser = argparse.ArgumentParser(
    description="""This script trains a separation neural network""")

  parser.add_argument("arch_file", metavar="arch-file", type=str,
                      help="DNN architecture file")
  parser.add_argument("gpu_id", metavar="gpu-id", type=int,
                      help="GPU ID")
  parser.add_argument("filelist", type=str,
                      help="Training sample filelist")
  parser.add_argument("dirout", type=str,
                      help="Output directory")

  parser.add_argument("--cv-filelist", type=str,
                      help="Cross validation sample filelist",
                      default="")
  parser.add_argument("--train-copy-location", type=str,
                      help="Copy training data here for I/O purposes",
                      default="")
  parser.add_argument("--batch-size", type=int,
                      help="Batch size",
                      default=100)
  parser.add_argument("--start-epoch", type=int,
                      help="Epoch to start from",
                      default=0)
  parser.add_argument("--num-epochs", type=int,
                      help="Total number of training epochs",
                      default=200)
  parser.add_argument("--learning-rate", type=float,
                      help="Learning rate",
                      default=0.001)

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


  print("loading datset")
  dataset = m.TrainSet2mix(args.filelist, args.train_copy_location)
  train_size = len(dataset)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collator, num_workers=1)
  if args.cv_filelist:
    cv_dataset = m.TrainSet2mix(args.cv_filelist)
    cv_size = len(cv_dataset)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, collate_fn=cv_dataset.collator)

  print("initializing model")
  model = m.EnhBLSTM(args.gpu_id)
  model.cuda()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  print("using lr="+str(args.learning_rate))

  if args.start_epoch == 0:
    torch.save(model.state_dict(), args.dirout+'/intermediate_models/model-init')
  else:
    model.load_state_dict(torch.load(args.dirout+'/intermediate_models/model-'+str(args.start_epoch).zfill(3), map_location=lambda storage, loc: storage.cuda()))

  epoch_losses = [[], []]
  epoch_cv_losses = [[], []]

  print("training")
  for epoch in range(args.start_epoch, args.num_epochs):
    epoch_loss = 0.0
    for i_batch, sample_batch in enumerate(dataloader):
      batch_dim = len(sample_batch)
      loss = m.compute_loss(model, sample_batch)/batch_dim
      epoch_loss += loss*batch_dim
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
      optimizer.step()

    if args.cv_filelist and epoch % 5 == 4:
      cv_loss = 0.0
      with torch.no_grad():
        for i_batch_cv, sample_batch_cv in enumerate(cv_dataloader):
          if i_batch_cv == 0:
            cv_loss += m.compute_loss(model, sample_batch_cv, args.dirout+'/plots/epoch'+str(epoch+1).zfill(3))
          else:
            cv_loss += m.compute_loss(model, sample_batch_cv)
      print("For epoch: "+str(epoch+1).zfill(3)+" cv set loss is: "+str(cv_loss.detach().cpu().numpy()/cv_size))
      epoch_cv_losses[0].append(epoch+1)
      epoch_cv_losses[1].append(cv_loss.detach().cpu().numpy()/cv_size)

    print("For epoch: "+str(epoch+1).zfill(3)+" loss is: "+str(epoch_loss.detach().cpu().numpy()/train_size))
    epoch_losses[0].append(epoch+1)
    epoch_losses[1].append(epoch_loss.detach().cpu().numpy()/train_size)
    if epoch % 5 == 4:
      print("Saving model for epoch "+str(epoch+1).zfill(3))
      torch.save(model.state_dict(), args.dirout+'/intermediate_models/model-'+str(epoch+1).zfill(3))
      qp.line(epoch_losses, epoch_cv_losses, args.dirout+'/plots/epoch'+str(epoch+1).zfill(3)+'/losses_'+str(args.start_epoch+1).zfill(3)+'-'+str(epoch+1).zfill(3)+'.png')
    sys.stdout.flush()

  torch.save(model.state_dict(), args.dirout+'/model-final')
  qp.line(epoch_losses, epoch_cv_losses, args.dirout+'/plots/losses_'+str(args.start_epoch+1).zfill(3)+'-'+str(args.num_epochs).zfill(3)+'.png')

if __name__ == '__main__':
  main()
