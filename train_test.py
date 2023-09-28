import wandb
import numpy as np
import utils
import torch
import os
from tqdm.auto import tqdm
import torch.nn.functional as F

def train_epoch(args, loader, data_info, data_bonds, data_charges, epoch, model, model_ema, 
                ema, device, dtype, optim, gradnorm_queue):
    model.train()
    loss_epoch = []
    loss_energy_epoch = []
    loss_force_epoch = []
    for i, data in enumerate(loader):
        x = data[0].to(device, dtype)
        forces = data[2].to(device, dtype)
        energies = data[1].to(device, dtype)
        h, bond, edge_index =utils.get_atom_bond_type(data_charges, data_bonds, args.batch_size, data_info, device, dtype)
        bs, n_nodes, _ = x.shape
        x = x.view(bs * n_nodes, -1)
        forces = forces.view(bs * n_nodes, -1)
        x, energies, forces = utils.data_normalize(data_info, device, dtype, x, energies, forces)
        optim.zero_grad()

        pred_energies, pred_forces = model(x, h, edge_index, bond)
        pred_forces = pred_forces.to(device, dtype)
        pred_energies = -pred_energies.to(device, dtype)
        
        # MSE损失训练会稳定些
        loss_func = torch.nn.L1Loss()
        energy_loss = loss_func(pred_energies, energies)
        force_loss = loss_func(pred_forces, forces)
        loss = energy_loss + force_loss

        loss.backward()
        grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        optim.step()
        ema.update_model_average(model_ema, model)

        loss_epoch.append(loss.item())
        loss_energy_epoch.append(energy_loss.item())
        loss_force_epoch.append(force_loss.item())
    loss_mean = np.mean(loss_epoch)
    print(f"\rTrain Epoch: {epoch}, Loss: {loss_mean:.5f}")
    wandb.log({'train epoch loss': loss_mean, 
               'train epoch energy loss': np.mean(loss_energy_epoch),
               'train epoch force loss': np.mean(loss_force_epoch),
               'epoch': epoch})

def validation(args, loader, data_info, data_bonds, data_charges, epoch, eval_model, device, dtype):
    eval_model.eval()
    loss_epoch = []
    loss_energy_epoch = []
    loss_force_epoch = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data[0].to(device, dtype)
            forces = data[2].to(device, dtype)
            energies = data[1].to(device, dtype)
            h, bond, edge_index =utils.get_atom_bond_type(data_charges, data_bonds, args.batch_size, data_info, device, dtype)
            bs, n_nodes, _ = x.shape
            x = x.view(bs * n_nodes, -1)
            forces = forces.view(bs * n_nodes, -1)
            x, _, __ = utils.data_normalize(data_info, device, dtype, x)

            pred_energies, pred_forces = eval_model(x, h, edge_index, bond)
            pred_forces = pred_forces.to(device, dtype)
            pred_energies = -pred_energies.to(device, dtype)
            x, pred_energies, pred_forces = utils.data_unnormalize(data_info, device, dtype, x, pred_energies, pred_forces)
            
            loss_func = torch.nn.L1Loss(reduction='sum')
            energy_loss = loss_func(pred_energies, energies)
            force_loss = loss_func(pred_forces, forces)
            loss = energy_loss + force_loss 
            loss_epoch.append(loss.item())
            loss_energy_epoch.append(energy_loss.item())
            loss_force_epoch.append(force_loss.item())
    return np.sum(loss_epoch), np.sum(loss_energy_epoch), np.sum(loss_force_epoch)

def test(batch_size, loader, data_info, data_bonds, data_charges, eval_model, device, dtype):
    eval_model.eval()
    forces, energies = [], []
    coords = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.to(device, dtype)
            h, bond, edge_index =utils.get_atom_bond_type(data_charges, data_bonds, batch_size, data_info, device, dtype)
            bs, n_nodes, _ = x.shape
            x = x.view(bs * n_nodes, -1)
            x, _, __ = utils.data_normalize(data_info, device, dtype, x)

            pred_energies, pred_forces = eval_model(x, h, edge_index, bond)
            pred_forces = pred_forces.to(device, dtype)
            pred_energies = -pred_energies.to(device, dtype)
            x, pred_energies, pred_forces = utils.data_unnormalize(data_info, device, dtype, x, pred_energies, pred_forces)

            pred_forces = pred_forces.view(bs, n_nodes, -1).to('cpu', torch.float64).numpy()
            pred_energies = pred_energies.squeeze().to('cpu', torch.float64).numpy()
            forces.append(pred_forces)
            energies.append(pred_energies)
            x = x.view(bs, n_nodes, -1).to('cpu', torch.float64).numpy()
            coords.append(x)
            
    forces = np.concatenate(forces, axis=0)
    energies = np.concatenate(energies, axis=0)
    coords = np.concatenate(coords, axis=0)
    return forces, energies, coords
