import torch
import numpy as np
import torch.nn.functional as F
# from egnn import Frame_EGNN

# class CG2310_Model(nn.Module):
#     def __init__(self, in_node_dim, in_bond_dim, hidden_dim, node_emb_dim=8, bond_emb_dim=4,
#                  out_energy_dim=1, out_force_dim=3, device='cpu', 
#                  act_fn=nn.SiLU(), n_layers=2, attention=True, 
#                  num_rbf=16, rbf_trainable=False, aggregation_method='sum'):
#         super().__init__()
#         self.egnn = Frame_EGNN(
#             in_node_dim=in_node_dim, in_bond_dim=in_bond_dim, hidden_dim=hidden_dim, node_emb_dim=node_emb_dim, 
#             bond_emb_dim=bond_emb_dim, out_energy_dim=out_energy_dim, out_force_dim=out_force_dim, device=device, 
#             act_fn=act_fn, n_layers=n_layers, attention=attention, num_rbf=num_rbf, rbf_trainable=rbf_trainable, 
#             aggregation_method=aggregation_method
#         )
#         self.device = device
#         self._edges_dict = {}

#     def forward(self, x, h, edge_index, bond):
#         raise NotImplementedError

#     def _forward(self, x, h, edge_index, bond):
#         bs, n_nodes, _ = x.shape

def data_normalize(data_info, device, dtype, x, energy=None, force=None):
    x_mean, x_std = data_info['coords_normalize'][0], data_info['coords_normalize'][1]
    x_mean = torch.Tensor(x_mean).unsqueeze(dim=0).to(device, dtype)
    x_std = torch.Tensor(x_std).unsqueeze(dim=0).to(device, dtype)
    x_norm = (x - x_mean) / x_std

    energy_norm, force_norm = None, None
    if energy is not None:
        energy_mean, energy_std = data_info['energy_normalize'][0], data_info['energy_normalize'][1]
        energy_mean = torch.Tensor([energy_mean]).to(device, dtype)
        energy_std = torch.Tensor([energy_std]).to(device, dtype)
        energy_norm = (energy - energy_mean) / energy_std

        force_mean, force_std = data_info['forces_normalize'][0], data_info['forces_normalize'][1]
        force_mean = torch.Tensor(force_mean).unsqueeze(dim=0).to(device, dtype)
        force_std = torch.Tensor(force_std).unsqueeze(dim=0).to(device, dtype)
        force_norm = (force - force_mean) / force_std

    return x_norm, energy_norm, force_norm

def data_unnormalize(data_info, device, dtype, x, energy, force):
    x_mean, x_std = data_info['coords_normalize'][0], data_info['coords_normalize'][1]
    x_mean = torch.Tensor(x_mean).unsqueeze(dim=0).to(device, dtype)
    x_std = torch.Tensor(x_std).unsqueeze(dim=0).to(device, dtype)
    x_unnorm = x * x_std + x_mean

    energy_mean, energy_std = data_info['energy_normalize'][0], data_info['energy_normalize'][1]
    energy_mean = torch.Tensor([energy_mean]).to(device, dtype)
    energy_std = torch.Tensor([energy_std]).to(device, dtype)
    energy_unnorm = energy * energy_std + energy_mean

    force_mean, force_std = data_info['forces_normalize'][0], data_info['forces_normalize'][1]
    force_mean = torch.Tensor(force_mean).unsqueeze(dim=0).to(device, dtype)
    force_std = torch.Tensor(force_std).unsqueeze(dim=0).to(device, dtype)
    force_unnorm = force * force_std +force_mean

    return x_unnorm, energy_unnorm, force_unnorm


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def save_model(model, path):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def get_atom_bond_type(data_charges, data_bonds, batch_size, data_info, device, dtype):
    atom_decoder = data_info['atom_decoder']
    bond_num = data_info['bond_num']
    n_nodes = data_charges.shape[0]
    rows, cols, bonds_type, charges = [], [], [], []
    for batch_idx in range(batch_size):
        for bond in data_bonds:
            rows.append(bond[0] - 1 + batch_idx * n_nodes)
            cols.append(bond[1] - 1 + batch_idx * n_nodes)
            bonds_type.append(bond[2] - 1)
        for charge in data_charges:
            charges.append(charge)
    
    edge_index = [torch.LongTensor(rows).to(device),
                torch.LongTensor(cols).to(device)]
    bonds_type = torch.Tensor(bonds_type).to(device, torch.int64)
    atoms_type = torch.Tensor([atom_decoder[charge] for charge in charges]).to(device, torch.int64)
    bond = F.one_hot(bonds_type, bond_num).to(device, dtype)
    h = F.one_hot(atoms_type, len(atom_decoder)).to(device, dtype)
    return h, bond, edge_index

class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new