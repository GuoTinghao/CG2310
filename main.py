import torch
import argparse
import time
import numpy as np
import copy
from datasets.data_loader import CG3210Dataset, CG2310DataLoader, CG3210TestDataset, \
                                CG2310TestDataLoader, total_data_info, load_split_data
from model.models import Frame_EGNN
import utils
import train_test
import os
import wandb

torch.set_num_threads(16)
parser = argparse.ArgumentParser()
#parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--dataset', type=str, default='rmd17_benzene')
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--ema_decay', type=float, default=0.9999)
parser.add_argument('--factor', type=float, default=0.8)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--wandb_project', type=str, default='CG2310')
parser.add_argument('--avoid_same', type=str, default='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
exp_name = args.dataset + ' hidden dim' + str(args.hidden_dim) + ' seed' + str(args.seed) + ' lr' + str(args.lr) + args.avoid_same
args.exp_name = exp_name
utils.set_seed(args.seed)
try:
    os.makedirs('outputs/' + args.exp_name)
except OSError:
    pass

data_info = total_data_info[args.dataset + '_new.npz']

data = np.load('./datasets/train/' + args.dataset + '_new.npz')
test_data = np.load('./datasets/test/' + args.dataset + '_new.npz')
test_dataset = CG3210TestDataset(test_data, device)
test_dataloader = CG2310TestDataLoader(test_dataset, 100)

data_charges = data['nuclear_charges']
data_bonds = data['bonds']

# Wandb args
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'name': args.exp_name, 'project': args.wandb_project, 'settings': wandb.Settings(_disable_stats=True), 
          'reinit': True, 'mode': mode, 'config': args}
wandb.init(**kwargs)

def main():
    model = Frame_EGNN(in_node_dim=len(data_info['atom_decoder']), 
                    in_bond_dim=data_info['bond_num'],
                    hidden_dim=args.hidden_dim,
                    batch_size=args.batch_size,
                    n_nodes=data_info['atom_num'])
    model = model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            factor=args.factor,
            patience=args.patience,
            min_lr=args.min_lr
        )
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    model_ema = copy.deepcopy(model)
    ema = utils.EMA(args.ema_decay)

    best_val_loss = 1e8
    is_ema = True
    for epoch in range(args.n_epochs):
        train_data, val_data = load_split_data(data, val_proportion=0.1)

        train_dataset = CG3210Dataset(train_data, device)
        #shuffle没有采用是因为每个epoch的load_split_data已经随机划分了数据集
        train_dataloader = CG2310DataLoader(train_dataset, args.batch_size)

        val_dataset = CG3210Dataset(val_data, device)
        val_dataloader = CG2310DataLoader(val_dataset, args.batch_size) 

        start_epoch = time.time()
        train_test.train_epoch(args, train_dataloader, data_info, data_bonds, data_charges, epoch, 
                               model, model_ema, ema, device, dtype, optim, gradnorm_queue)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        val_loss_ema, val_loss_energy_ema, val_loss_force_ema = \
            train_test.validation(args, val_dataloader, data_info, data_bonds, data_charges, epoch, model_ema, device, dtype)
        val_loss, val_loss_energy, val_loss_force = \
            train_test.validation(args, val_dataloader, data_info, data_bonds, data_charges, epoch, model_ema, device, dtype)
        if val_loss_ema < val_loss:
            val_loss = val_loss_ema
            val_loss_energy = val_loss_energy_ema
            val_loss_force = val_loss_force_ema
            is_ema = True
        else:
            is_ema = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_ema:
                utils.save_model(model_ema, 'outputs/%s/model.npy' % args.exp_name)
            else:
                utils.save_model(model, 'outputs/%s/model.npy' % args.exp_name)


        print(f"\rVal Epoch: {epoch}, Val Loss: {val_loss:.5f}, Best Val Loss: {best_val_loss:.5f}")
        wandb.log({'val epoch loss': val_loss, 
                   'Best val loss': best_val_loss, 
                   'val epoch energy loss': val_loss_energy, 
                   'val epoch force loss': val_loss_force, 
                   'epoch': epoch})
        
        scheduler.step(val_loss)
    print(args.dataset + "finish!")

if __name__ == "__main__":
    main()