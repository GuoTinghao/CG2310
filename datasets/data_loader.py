import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

total_data_info = {
    'rmd17_aspirin_new.npz': {'atom_num': 21,
                            'atom_decoder': {1:0, 6:1, 8:2},
                            'bond_num': 2,
                            'bias_energy': -406274.58646878117,
                            'scale_force': 1e2,
                            'coords_normalize': [[-0.0028897089380952373, 0.18125123638904764, -0.07988123513999999],
                                                [2.237121625275375, 1.5817961493059836, 1.169947653256711]],
                            'forces_normalize': [[3.8962691775168426e-09, 6.776120551015603e-10, 3.049254103649175e-09],
                                                [30.328098654691335, 30.012773847357522, 27.582380263469027]],
                            'energy_normalize': [-406274.58646878117, 6.1113810303915175],
                          },

    'rmd17_azobenzene_new.npz': {'atom_num': 24,
                            'atom_decoder': {1:0, 6:1, 7:2},
                            'bond_num': 2,
                            'bias_energy': -358670.61259150563,
                            'scale_force': 1e2,
                            'coords_normalize': [[0.00039409754125000146, -0.0010452054704166657, 0.0010361213699999986],
                                                [3.440884807943757, 0.986573473953345, 1.0663407776804914]],
                            'forces_normalize': [[2.371642051635092e-09, -2.96455300308196e-10, 2.223414451070956e-09],
                                                [34.606114431209186, 26.01305308650652, 28.064679738853734]],
                            'energy_normalize': [-358670.61259150563, 6.304535692375925],
                          },

    'rmd17_benzene_new.npz': {'atom_num': 12,
                            'atom_decoder': {1:0, 6:1},
                            'bond_num': 2,
                            'bias_energy': -145431.17004910586,
                            'scale_force': 1e2,
                            'coords_normalize': [[-102.20414641013083, 113.95602342695, -95.13877019649833],
                                                [49.5286986824308, 55.21154095121903, 46.09614984716578]],
                            'forces_normalize': [[9.881839021138225e-11, -5.731468354142066e-09, 4.6444657947712165e-09],
                                                [24.70456637384294, 19.61622408457957, 18.78927297298587]],
                            'energy_normalize': [-145431.17004910586, 2.430916980181677],
                          },

    'rmd17_ethanol_new.npz': {'atom_num': 9,
                            'atom_decoder': {1:0, 6:1, 8:2},
                            'bond_num': 1,
                            'bias_energy': -97076.07799709898,
                            'scale_force': 1e2,
                            'coords_normalize': [[-0.20783096041777777, 0.15997738340888887, -0.030880000579999994],
                                                [0.980767069076195, 0.9001430473773945, 0.8400021003094117]],
                            'forces_normalize': [[6.587894618961501e-10, 1.1842378929335004e-17, -6.983168381522217e-09],
                                                [26.379241779078754, 27.762184394190417, 27.958488526089553]],
                            'energy_normalize': [-97076.07799709898, 4.3084547695721405],
                          },

    'rmd17_malonaldehyde_new.npz': {'atom_num': 9,
                            'atom_decoder': {1:0, 6:1, 8:2},
                            'bond_num': 2,
                            'bias_energy': -167305.12048617186,
                            'scale_force': 1e2,
                            'coords_normalize': [[-0.009529496090000002, -0.001960989402222222, 0.0016266173922222223],
                                                [0.9589764011648944, 0.8974214838973004, 1.1414395088804075]],
                            'forces_normalize': [[2.503399935941767e-09, -3.4257051895439063e-09, 3.1621895186301824e-09],
                                                [30.735826999951072, 30.152919322275075, 30.456522232887643]],
                            'energy_normalize': [-167305.12048617186, 4.3047195199546024],
                          },

    'rmd17_naphthalene_new.npz': {'atom_num': 18,
                            'atom_decoder': {1:0, 6:1},
                            'bond_num': 2,
                            'bias_energy': -241637.5863391555,
                            'scale_force': 1e2,
                            'coords_normalize': [[0.0003136672966666599, 0.0003527529288888872, -0.0007644554822222224],
                                                [2.124533540950588, 1.5117156791159274, 0.3073601590384616]],
                            'forces_normalize': [[-1.5810947723417081e-09, 3.6892210357248385e-09, -1.8446105018752078e-09],
                                                [35.04200948726983, 35.74176736119866, 15.711405707901381]],
                            'energy_normalize': [-241637.5863391555, 5.433610025288334],
                          },

    'rmd17_paracetamol_new.npz': {'atom_num': 20,
                            'atom_decoder': {1:0, 6:1, 7:2, 8:3},
                            'bond_num': 2,
                            'bias_energy': -322826.03226281697,
                            'scale_force': 1e2,
                            'coords_normalize': [[4.541835446727, -0.6761569320440001, -0.15320162862199999],
                                                [2.7064652640284, 1.0282210454076794, 0.9540996791955139]],
                            'forces_normalize': [[-1.3636942121308948e-09, 3.3202989541791794e-09, -2.3716425197051193e-10],
                                                [31.367399510678972, 27.702998639146397, 28.763761972673517]],
                            'energy_normalize': [-322826.03226281697, 6.2995559114991275],
                          },

    'rmd17_salicylic_new.npz': {'atom_num': 16,
                            'atom_decoder': {1:0, 6:1, 8:2},
                            'bond_num': 2,
                            'bias_energy': -310680.4890705824,
                            'scale_force': 1e2,
                            'coords_normalize': [[-0.340099394449375, -0.190264556119375, -0.0070615040100000005],
                                                [1.9823884444113309, 1.4607123499652117, 0.2503944842780207]],
                            'forces_normalize': [[5.187967346831357e-10, -7.411373181831493e-11, -2.964552616113725e-09],
                                                [35.484725399366404, 34.68321147807169, 15.308273715177778]],
                            'energy_normalize': [-310680.4890705824, 5.804618815798429],
                          },

    'rmd17_toluene_new.npz': {'atom_num': 15,
                            'atom_decoder': {1:0, 6:1},
                            'bond_num': 2,
                            'bias_energy': -170036.76184642038,
                            'scale_force': 1e2,
                            'coords_normalize': [[0.22896167100999998, -0.016866861541333336, 0.01408640696133333],
                                                [1.870800754968105, 0.9666063498673245, 0.9725558065425034]],
                            'forces_normalize': [[-3.952736847168126e-09, 5.533831201868604e-10, 2.1344778896074484e-09],
                                                [30.997789646855846, 27.12514003883341, 27.528333542799345]],
                            'energy_normalize': [-170036.76184642038, 5.074269847451408],
                          },

    'rmd17_uracil_new.npz': {'atom_num': 12,
                            'atom_decoder': {1:0, 6:1, 7:2, 8:3},
                            'bond_num': 2,
                            'bias_energy': -259813.37373236,
                            'scale_force': 1e2,
                            'coords_normalize': [[0.2910221477458333, -0.21365863446083336, -0.0092580689],
                                                [1.4304962286137934, 1.424705502170492, 0.22061581961572396]],
                            'forces_normalize': [[-2.3716420122591824e-09, 2.3716420643656495e-09, 9.881842129762693e-11],
                                                [35.20877139191942, 38.07991981458343, 16.16471582927557]],
                            'energy_normalize': [-259813.37373236, 5.129788472557705],
                          },
}

def load_split_data(data, val_proportion=0.1):
    data_coords = data['coords']
    data_forces = data['forces']
    data_energies = data['energies']
    
    perm = np.random.permutation(data_coords.shape[0]).astype('int32')
    data_list_coords = [data_coords[i] for i in perm]
    data_list_forces = [data_forces[i] for i in perm]
    data_list_energies = [data_energies[i] for i in perm]


    val_index = int(data_coords.shape[0] * val_proportion)
    val_data_coords, train_data_coords = data_list_coords[:val_index], data_list_coords[val_index:]
    val_data_forces, train_data_forces = data_list_forces[:val_index], data_list_forces[val_index:]
    val_data_energies, train_data_energies = data_list_energies[:val_index], data_list_energies[val_index:]
    train_data = {'coords': train_data_coords,
                  'forces': train_data_forces,
                  'energies': train_data_energies}
    val_data = {'coords': val_data_coords,
                  'forces': val_data_forces,
                  'energies': val_data_energies}
    return train_data, val_data

class CG3210Dataset(Dataset):
    def __init__(self, data_list, device):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = data_list
        self.device = device

    def __len__(self):
        return len(self.data_list['coords'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        coords = torch.from_numpy(self.data_list['coords'][idx]).to(self.device)
        energies = torch.Tensor([self.data_list['energies'][idx]]).to(self.device)
        forces = torch.from_numpy(self.data_list['forces'][idx]).to(self.device)
        return [coords, energies, forces]
    
def collate_fn(batch):
    device = batch[0][0][0].device
    batch_coords, batch_energies, batch_forces = [], [], []
    for data in batch:
        batch_coords.append(data[0])
        batch_energies.append(data[1])
        batch_forces.append(data[2])
    batch_coords = torch.stack(batch_coords).to(device)
    batch_energies = torch.stack(batch_energies).to(device)
    batch_forces = torch.stack(batch_forces).to(device)
    return [batch_coords, batch_energies, batch_forces]


class CG2310DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn)

class CG3210TestDataset(Dataset):
    def __init__(self, data_list, device):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = data_list
        self.device = device

    def __len__(self):
        return len(self.data_list['coords'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        coords = torch.from_numpy(self.data_list['coords'][idx]).to(self.device)
        return coords
    
def test_collate_fn(batch):
    device = batch[0][0].device
    batch_coords = []
    for data in batch:
        batch_coords.append(data)
    batch_coords = torch.stack(batch_coords).to(device)
    return batch_coords


class CG2310TestDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size, collate_fn=test_collate_fn)