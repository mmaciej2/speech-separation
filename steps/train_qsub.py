#!/usr/bin/env python3

import argparse
import sys
import os
import datetime
import numpy as np

sys.path.append('tools')
import plot

import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

sys.path.append('archs')

def get_args():
  parser = argparse.ArgumentParser(
    description="""This script trains a separation neural network""")

  parser.add_argument("arch_file", metavar="arch-file", type=str,
                      help="DNN architecture file")
  parser.add_argument("gpu_id", metavar="gpu-id", type=int,
                      help="GPU ID")
  parser.add_argument("data_dir", metavar="data-dir", type=str,
                      help="Training data directory")
  parser.add_argument("dirout", type=str,
                      help="Output directory")

  parser.add_argument("--cv-data-dir", type=str,
                      help="Cross validation data directory",
                      default="")
  parser.add_argument("--train-copy-location", type=str,
                      help="Copy training data here for I/O purposes",
                      default="")
  parser.add_argument("--n-debug", type=int,
                      help="Truncate dataset to n files for debug purposes",
                      default=-1)
  parser.add_argument("--model-config", type=str,
                      help="Config file for DNN",
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
  parser.add_argument("--grad-clip-norm", type=float,
                      help="l2 norm for gradient clipping",
                      default=5)
  parser.add_argument("--LR-decay-patience", type=int,
                      help="Number of non-decreasing lossses before decay",
                      default=3)
  parser.add_argument("--LR-decay-factor", type=float,
                      help="LR decay factor",
                      default=0.5)

  args = parser.parse_args()
  return args

def load_losses(filename, loss_array):
  with open(filename, 'r') as lossF:
    for line in lossF:
      split = line.rstrip().split()
      loss_array[0].append(int(split[0]))
      loss_array[1].append(np.float(split[1]))

class LRDecay(optim.lr_scheduler.ReduceLROnPlateau):
  def __init__(self, *args, **kwargs):
    super(LRDecay, self).__init__(*args, **kwargs)

  def step(self, metrics, epoch=None):
    current = float(metrics)
    if epoch is None:
      epoch = self.last_epoch + 1
    else:
      warnings.warn(EPOCH_DEPRECATION_WARNING, DeprecationWarning)
    self.last_epoch = epoch

    if self.is_better(current, self.best):
      self.best = current
      self.num_bad_epochs = 0
    else:
      self.num_bad_epochs += 1

    if self.in_cooldown:
      self.cooldown_counter -= 1
      self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

    if self.num_bad_epochs >= self.patience:
      self._reduce_lr(epoch)
      self.cooldown_counter = self.cooldown
      self.num_bad_epochs = 0
      return_val = True
    else:
      return_val = False

    self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    return return_val

def main():
  args = get_args()
  global m
  print("Using "+args.arch_file+" DNN architecture")
  m = __import__(args.arch_file)

  print("Using GPU", args.gpu_id)
  torch.cuda.set_device(0)
  tmp = torch.ByteTensor([0])
  tmp.cuda()

  int_model_dir = args.dirout+'/intermediate_models/'
  plot_dir = args.dirout+'/stats_tr/plots/'
  loss_file = args.dirout+'/stats_tr/train_loss.txt'
  cv_loss_file = args.dirout+'/stats_tr/cv_loss.txt'


  print("loading datset")
  dataset = m.TrainSet(args.data_dir, args.train_copy_location)
  if args.n_debug > 0:
    datasubset = Subset(dataset, range(args.n_debug))
    dataloader = DataLoader(datasubset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collator)
  else:
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collator)
  if args.cv_data_dir:
    cv_dataset = m.ValidationSet(args.cv_data_dir, args.train_copy_location)
    if args.n_debug > 0:
      cv_datasubset = Subset(cv_dataset, range(args.n_debug))
      cv_dataloader = DataLoader(cv_datasubset, batch_size=args.batch_size, collate_fn=cv_dataset.collator)
    else:
      cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, collate_fn=cv_dataset.collator)

  print("initializing model")
  if args.model_config:
    kwargs = dict()
    for line in open(args.model_config):
      kwargs[line.split('=')[0]] = line.rstrip().split('=')[1]
    model = m.SepDNN(args.gpu_id, **kwargs)
  else:
    model = m.SepDNN(args.gpu_id)
  model.cuda()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  print("using lr="+str(args.learning_rate))
  lr_scheduler = LRDecay(optimizer, factor=args.LR_decay_factor, patience=args.LR_decay_patience, verbose=True, threshold=1e-4)

  epoch_losses = [[], []]
  epoch_cv_losses = [[], []]
  lossF = open(loss_file, 'a')
  if args.cv_data_dir:
    cv_lossF = open(cv_loss_file, 'a')

  if args.start_epoch == 0:
    torch.save(model.state_dict(), args.dirout+'/best.mdl')
  else:
    model.load_state_dict(torch.load(args.dirout+'/best.mdl', map_location=lambda storage, loc: storage.cuda()))
    load_losses(loss_file, epoch_losses)
    if args.cv_data_dir:
      load_losses(cv_loss_file, epoch_cv_losses)

  print("training")
  print("---", datetime.datetime.now(), "------------------------------------")
  best_loss = 1e12
  for epoch in range(args.start_epoch, args.num_epochs):
    epoch_loss = 0.0
    epoch_norm = 0
    for i_batch, sample_batch in enumerate(dataloader):
      loss, norm = m.compute_loss(model, epoch, sample_batch)
      epoch_loss += loss.detach().cpu().numpy() * norm.detach().cpu().numpy()
      epoch_norm += norm.detach().cpu().numpy()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
      optimizer.step()

      if epoch == 0 and  np.round(np.log2(i_batch+1)) == np.log2(i_batch+1):
        print("Epoch", str(epoch+1).zfill(3), "iteration", i_batch+1, "has loss", loss.detach().cpu().numpy())
        print("---", datetime.datetime.now(), "------------------------------------")
        sys.stdout.flush()

    if args.cv_data_dir:
      cv_epoch_loss = 0.0
      cv_epoch_norm = 0
      model.eval()
      with torch.no_grad():
        for i_batch_cv, sample_batch_cv in enumerate(cv_dataloader):
          if i_batch_cv == 0:
            cv_loss, cv_norm = m.compute_cv_loss(model, epoch, sample_batch_cv, plot_dir+'epoch'+str(epoch+1).zfill(3))
          else:
            cv_loss, cv_norm = m.compute_cv_loss(model, epoch, sample_batch_cv)
          cv_epoch_loss += cv_loss.detach().cpu().numpy() * cv_norm.detach().cpu().numpy()
          cv_epoch_norm += cv_norm.detach().cpu().numpy()
      model.train()
      print("For epoch: "+str(epoch+1).zfill(3)+" validation loss is: "+str(cv_epoch_loss/cv_epoch_norm))
      cv_lossF.write(str(epoch+1).zfill(3)+' '+str(cv_epoch_loss/cv_epoch_norm)+'\n')
      cv_lossF.flush()
      epoch_cv_losses[0].append(epoch+1)
      epoch_cv_losses[1].append(cv_epoch_loss/cv_epoch_norm)

    print("For epoch: "+str(epoch+1).zfill(3)+" training loss is: "+str(epoch_loss/epoch_norm))
    lossF.write(str(epoch+1).zfill(3)+' '+str(epoch_loss/epoch_norm)+'\n')
    lossF.flush()
    epoch_losses[0].append(epoch+1)
    epoch_losses[1].append(epoch_loss/epoch_norm)

    if args.cv_data_dir:
      if cv_epoch_loss/cv_epoch_norm < best_loss:
        best_loss = cv_epoch_loss/cv_epoch_norm
        print("Saving model for epoch "+str(epoch+1).zfill(3))
        torch.save(model.state_dict(), args.dirout+'/best.mdl')
      else:
        print("Model has not improved.")
      if lr_scheduler.step(cv_epoch_loss/cv_epoch_norm):
        model.load_state_dict(torch.load(args.dirout+'/best.mdl', map_location=lambda storage, loc: storage.cuda()))
    else:
      if epoch_loss/epoch_norm < best_loss:
        best_loss = epoch_loss/epoch_norm
        print("Saving model for epoch "+str(epoch+1).zfill(3))
        torch.save(model.state_dict(), args.dirout+'/best.mdl')
      else:
        print("Model has not improved.")
      if lr_scheduler.step(epoch_loss/epoch_norm):
        model.load_state_dict(torch.load(args.dirout+'/best.mdl', map_location=lambda storage, loc: storage.cuda()))

    if epoch % 25 == 24:
      print("Saving intermediate model at epoch "+str(epoch+1).zfill(3))
      torch.save(model.state_dict(), int_model_dir+str(epoch+1).zfill(3)+'.mdl')
    print("---", datetime.datetime.now(), "------------------------------------")
    sys.stdout.flush()

  torch.save(model.state_dict(), int_model_dir+'final.mdl')
  plot.plot_loss(epoch_losses, epoch_cv_losses, plot_dir+'Loss_'+str(epoch_losses[0][0]).zfill(3)+'-'+str(args.num_epochs).zfill(3)+'.png')

if __name__ == '__main__':
  main()
