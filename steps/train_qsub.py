#!/usr/bin/env python3

import torch
import argparse
import sys
import os
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.append('archs')

sys.path.append('tools')
import plot

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
  dataset = m.TrainSet(args.data_dir, args.train_copy_location)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collator, num_workers=1)
  if args.cv_data_dir:
    cv_dataset = m.TrainSet(args.cv_data_dir)
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

  epoch_losses = [[], []]
  epoch_cv_losses = [[], []]
  lossF = open(loss_file, 'a')
  if args.cv_data_dir:
    cv_lossF = open(cv_loss_file, 'a')

  if args.start_epoch == 0:
    torch.save(model.state_dict(), int_model_dir+'init.mdl')
  else:
    model.load_state_dict(torch.load(int_model_dir+str(args.start_epoch).zfill(3)+'.mdl', map_location=lambda storage, loc: storage.cuda()))
    load_losses(loss_file, epoch_losses)
    if args.cv_data_dir:
      load_losses(cv_loss_file, epoch_cv_losses)

  print("training")
  for epoch in range(args.start_epoch, args.num_epochs):
    epoch_loss = 0.0
    epoch_norm = 0
    for i_batch, sample_batch in enumerate(dataloader):
      loss, norm = m.compute_loss(model, sample_batch)
      epoch_loss += loss.detach().cpu().numpy() * norm.detach().cpu().numpy()
      epoch_norm += norm.detach().cpu().numpy()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
      optimizer.step()

    if args.cv_data_dir and epoch % 5 == 4:
      cv_epoch_loss = 0.0
      cv_epoch_norm = 0
      with torch.no_grad():
        for i_batch_cv, sample_batch_cv in enumerate(cv_dataloader):
          if i_batch_cv == 0:
            cv_loss, cv_norm = m.compute_cv_loss(model, sample_batch_cv, plot_dir+'epoch'+str(epoch+1).zfill(3))
          else:
            cv_loss, cv_norm = m.compute_cv_loss(model, sample_batch_cv)
          cv_epoch_loss += cv_loss.detach().cpu().numpy() * cv_norm.detach().cpu().numpy()
          cv_epoch_norm += cv_norm.detach().cpu().numpy()
      print("For epoch: "+str(epoch+1).zfill(3)+" cv set loss is: "+str(cv_epoch_loss/cv_epoch_norm))
      cv_lossF.write(str(epoch+1).zfill(3)+' '+str(cv_epoch_loss/cv_epoch_norm)+'\n')
      cv_lossF.flush()
      epoch_cv_losses[0].append(epoch+1)
      epoch_cv_losses[1].append(cv_epoch_loss/cv_epoch_norm)

    print("For epoch: "+str(epoch+1).zfill(3)+" loss is: "+str(epoch_loss/epoch_norm))
    lossF.write(str(epoch+1).zfill(3)+' '+str(epoch_loss/epoch_norm)+'\n')
    lossF.flush()
    epoch_losses[0].append(epoch+1)
    epoch_losses[1].append(epoch_loss/epoch_norm)
    if epoch % 5 == 4:
      print("Saving model for epoch "+str(epoch+1).zfill(3))
      torch.save(model.state_dict(), int_model_dir+str(epoch+1).zfill(3)+'.mdl')
      os.system("mkdir -p "+plot_dir+'epoch'+str(epoch+1).zfill(3))
      plot.plot_loss(epoch_losses, epoch_cv_losses, plot_dir+'epoch'+str(epoch+1).zfill(3)+'/Loss_'+str(epoch_losses[0][0]).zfill(3)+'-'+str(epoch+1).zfill(3)+'.png')
    sys.stdout.flush()

  torch.save(model.state_dict(), args.dirout+'/final.mdl')
  plot.plot_loss(epoch_losses, epoch_cv_losses, plot_dir+'Loss_'+str(epoch_losses[0][0]).zfill(3)+'-'+str(args.num_epochs).zfill(3)+'.png')

if __name__ == '__main__':
  main()
