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

sys.path.append('archs')

import quick_plot as qp

def get_args():
  parser = argparse.ArgumentParser(
    description="""This script trains a separation neural network""")

  parser.add_argument("arch_file", metavar="arch-file", type=str,
                      help="DNN architecture file")
  parser.add_argument("gpu_id", metavar="gpu-id", type=int,
                      help="GPU ID")
  parser.add_argument("feats", type=str,
                      help="Training sample features file")
  parser.add_argument("dirout", type=str,
                      help="Output directory")

  parser.add_argument("--cv-feats", type=str,
                      help="Cross validation sample features file",
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

def load_losses(filename, loss_array):
  with open(filename, 'r') as lossF:
    for line in lossF:
      split = line.rstrip().split()
      loss_array[0].append(int(split[0]))
      loss_array[1].append(np.float(split[1]))

def main():
  args = get_args()
  global m
  print("Using "+args.arch_file+" DNN architecture")
  m = __import__(args.arch_file)

  print("Using GPU", args.gpu_id)
  torch.cuda.set_device(args.gpu_id)
  tmp = torch.ByteTensor([0])
  tmp.cuda()

  int_model_dir = args.dirout+'/intermediate_models/'
  plot_dir = args.dirout+'/train_stats/plots/'
  loss_file = args.dirout+'/train_stats/train_loss.txt'
  cv_loss_file = args.dirout+'/train_stats/cv_loss.txt'


  print("loading datset")
  dataset = m.TrainSet(args.feats, args.train_copy_location)
  train_size = len(dataset)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collator, num_workers=1)
  if args.cv_feats:
    cv_dataset = m.TrainSet(args.cv_feats)
    cv_size = len(cv_dataset)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, collate_fn=cv_dataset.collator)

  print("initializing model")
  model = m.EnhBLSTM(args.gpu_id)
  model.cuda()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  print("using lr="+str(args.learning_rate))

  epoch_losses = [[], []]
  epoch_cv_losses = [[], []]
  lossF = open(loss_file, 'a')
  if args.cv_feats:
    cv_lossF = open(cv_loss_file, 'a')

  if args.start_epoch == 0:
    torch.save(model.state_dict(), int_model_dir+'init.mdl')
  else:
    model.load_state_dict(torch.load(int_model_dir+str(args.start_epoch).zfill(3)+'.mdl', map_location=lambda storage, loc: storage.cuda()))
    load_losses(loss_file, epoch_losses)
    if args.cv_feats:
      load_losses(cv_loss_file, epoch_cv_losses)

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

    if args.cv_feats and epoch % 5 == 4:
      cv_loss = 0.0
      with torch.no_grad():
        for i_batch_cv, sample_batch_cv in enumerate(cv_dataloader):
          if i_batch_cv == 0:
            cv_loss += m.compute_loss(model, sample_batch_cv, plot_dir+'epoch'+str(epoch+1).zfill(3))
          else:
            cv_loss += m.compute_loss(model, sample_batch_cv)
      print("For epoch: "+str(epoch+1).zfill(3)+" cv set loss is: "+str(cv_loss.detach().cpu().numpy()/cv_size))
      cv_lossF.write(str(epoch+1).zfill(3)+' '+str(cv_loss.detach().cpu().numpy()/cv_size)+'\n')
      epoch_cv_losses[0].append(epoch+1)
      epoch_cv_losses[1].append(cv_loss.detach().cpu().numpy()/cv_size)

    print("For epoch: "+str(epoch+1).zfill(3)+" loss is: "+str(epoch_loss.detach().cpu().numpy()/train_size))
    lossF.write(str(epoch+1).zfill(3)+' '+str(epoch_loss.detach().cpu().numpy()/train_size)+'\n')
    epoch_losses[0].append(epoch+1)
    epoch_losses[1].append(epoch_loss.detach().cpu().numpy()/train_size)
    if epoch % 5 == 4:
      print("Saving model for epoch "+str(epoch+1).zfill(3))
      torch.save(model.state_dict(), int_model_dir+str(epoch+1).zfill(3)+'.mdl')
      os.system("mkdir -p "+plot_dir+'epoch'+str(epoch+1).zfill(3))
      qp.line(epoch_losses, epoch_cv_losses, plot_dir+'epoch'+str(epoch+1).zfill(3)+'/losses_'+str(epoch_losses[0][0]).zfill(3)+'-'+str(epoch+1).zfill(3)+'.png')
    sys.stdout.flush()

  torch.save(model.state_dict(), args.dirout+'/final.mdl')
  qp.line(epoch_losses, epoch_cv_losses, plot_dir+'losses_'+str(epoch_losses[0][0]).zfill(3)+'-'+str(args.num_epochs).zfill(3)+'.png')

if __name__ == '__main__':
  main()
