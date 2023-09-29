import torch
import argparse
import numpy as np
from datasets.data_loader import CG3210TestDataset, CG2310TestDataLoader, total_data_info
from model.models import Frame_EGNN
import utils
import train_test


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
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
exp_name = args.dataset + ' hidden dim' + str(args.hidden_dim) + ' seed' + str(args.seed) + ' lr' + str(args.lr)
args.exp_name = exp_name
utils.set_seed(args.seed)

data_info = total_data_info[args.dataset + '_new.npz']

test_data = np.load('./datasets/test/' + args.dataset + '_new.npz')
test_dataset = CG3210TestDataset(test_data, device)
test_dataloader = CG2310TestDataLoader(test_dataset, 100)

data_charges = test_data['nuclear_charges']
data_bonds = test_data['bonds']

def main():
    model = Frame_EGNN(in_node_dim=len(data_info['atom_decoder']), 
                    in_bond_dim=data_info['bond_num'],
                    hidden_dim=args.hidden_dim,
                    batch_size=args.batch_size,
                    n_nodes=data_info['atom_num'])
    model = model.to(device)
    model_state_dict = torch.load('outputs/%s/model.npy' % args.exp_name)
    model.load_state_dict(model_state_dict)

    test_forces, test_energies = train_test.test(100, test_dataloader, data_info, data_bonds, 
                                                 data_charges, model, device, dtype)
    np.savez('./datasets/test/' + args.dataset + '_label.npz', 
             energies=test_energies, 
             forces=test_forces)
    print(args.dataset + "finish!")

if __name__ == "__main__":
    main()