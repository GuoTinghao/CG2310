import torch.nn as nn
import torch

class GaussianSmearing(nn.Module):
    """Smears a distance distribution by a Gaussian function."""
    def __init__(self,
                 num_rbf=50,
                 rbound_upper=5.0,
                 rbound_lower=0.0,
                 rbf_trainable=False):
        super(GaussianSmearing, self).__init__()

        self.o = torch.linspace(rbound_lower, rbound_upper, num_rbf)
        margin = self.o[1] - self.o[0] if num_rbf > 1 else torch.Tensor([rbound_upper - rbound_lower])
        self.c = -0.5 / margin ** 2
        if rbf_trainable:
            self.register_parameter("coeff", nn.parameter.Parameter(self.c))
            self.register_parameter("offset", nn.parameter.Parameter(self.o))
        else:
            self.register_buffer("offset", self.o)
            self.register_buffer("coeff", self.c)

    def forward(self, dist):
        '''
        dist (B, 1)
        '''
        return torch.exp(self.coeff * torch.pow(dist - self.offset, 2))
    
class GCL(nn.Module):
    def __init__(self, in_node_dim, in_bond_dim, hidden_dim, out_node_dim=None, out_bond_dim=None, dis_dim=2,
                 aggregation_method='sum', act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(in_bond_dim + dis_dim + in_node_dim * 2),
            nn.Linear(in_bond_dim + dis_dim + in_node_dim * 2, hidden_dim * 2),
            act_fn,
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn)
        
        self.bond_mlp = nn.Sequential(
            nn.LayerNorm(in_bond_dim + hidden_dim),
            nn.Linear(in_bond_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_bond_dim),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + in_node_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_node_dim))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
            
    def edge_model(self, source, target, bond, dis_emb):
        m_ij = torch.cat([source, target, bond, dis_emb], dim=-1)
        m_ij = self.edge_mlp(m_ij)

        if self.attention:
            att_val = self.att_mlp(m_ij)
            m_ij = m_ij * att_val
        return m_ij
    
    def bond_model(self, bond, m_ij):
        agg = torch.cat([bond, m_ij], dim=-1)
        out = self.bond_mlp(agg)
        return out

    def node_model(self, h, agg):
        agg = torch.cat([h, agg], dim=-1)
        out = self.node_mlp(agg)
        return out

    def forward(self, dis_emb, h, edge_index, bond):
        row, col = edge_index
        m_ij = self.edge_model(h[row], h[col], bond, dis_emb)
        bond = self.bond_model(bond, m_ij)

        agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0),
                                   aggregation_method=self.aggregation_method)
        h = self.node_model(h, agg)
        return h, bond

class energy_block(nn.Module):
    def __init__(self, in_node_dim, in_bond_dim, hidden_dim, dis_dim,
                 batch_size=100, n_nodes=24, out_energy_dim=1,
                 aggregation_method='sum',act_fn=nn.SiLU()):
        super(energy_block, self).__init__()
        self.aggregation_method = aggregation_method
        self.batch_size = batch_size
        self.n_nodes = n_nodes

        self.message_mlp = nn.Sequential(
            nn.Linear(in_bond_dim + in_node_dim * 2 + 9 + dis_dim, hidden_dim * 2),
            act_fn,
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim))
        
        self.node_energy_hidden = nn.Sequential(
            nn.Linear(hidden_dim + in_node_dim, hidden_dim),
            act_fn)
        
        layer = nn.Linear(hidden_dim, out_energy_dim, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.node_energy_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim),
            layer
        )
        
        pooling = nn.Linear(n_nodes, out_energy_dim, bias=False)
        torch.nn.init.xavier_uniform_(pooling.weight, gain=0.001)
        self.total_energy_pooling = nn.Sequential(
            pooling,
            act_fn,
        )

        self.energy_attn = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
        
    def message_model(self, source, target, frame_feats, bond, dis_emb):
        m_ij = torch.cat([source, target, frame_feats, bond, dis_emb], dim=-1)
        m_ij = self.message_mlp(m_ij)

        return m_ij

    def node_model(self, h, agg):
        agg = torch.cat([h, agg], dim=-1)
        out = self.node_energy_hidden(agg)
        return out
    
    def energy_model(self, node_energy_hidden):
        node_energy_attn = self.energy_attn(node_energy_hidden)
        node_energy_hidden = node_energy_attn * node_energy_hidden

        node_energy = self.node_energy_pred(node_energy_hidden)
        return node_energy
    
    def forward(self, x, h, edge_index, bond, dis_emb):
        row, col = edge_index
        F_ij = localize(x, edge_index)
        frame_i = (x[row].unsqueeze(1) @ F_ij).squeeze(1)
        frame_j = (x[col].unsqueeze(1) @ F_ij).squeeze(1)
        frame_radial = torch.sum((frame_i - frame_j) ** 2, 1, keepdim=True)
        frame_cos = torch.clamp(nn.functional.cosine_similarity(frame_i, frame_j), min=-1, max=1).unsqueeze(-1)
        frame_sin = torch.sqrt(1 - frame_cos ** 2 + 1e-30)
        frame_feats = torch.cat([frame_i, frame_j, frame_radial, frame_cos, frame_sin], dim=-1)
        m_ij = self.message_model(h[row], h[col], frame_feats, bond, dis_emb)

        agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0),
                                   aggregation_method=self.aggregation_method)
        node_energy_hidden = self.node_model(h, agg)
        node_energy = self.energy_model(node_energy_hidden)
        node_energy = node_energy.view(self.batch_size, self.n_nodes)
        total_energy = self.total_energy_pooling(node_energy)
        return total_energy, node_energy_hidden


class force_block(nn.Module):
    def __init__(self, hidden_dim, out_force_dim=3,
                 aggregation_method='sum',act_fn=nn.SiLU()):
        super(force_block, self).__init__()
        self.aggregation_method = aggregation_method

        layer = nn.Linear(hidden_dim, out_force_dim, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.force_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim),
            layer
        )
    
    def forward(self, x, node_energy_hidden, edge_index):
        row, col = edge_index
        F_ij = localize(x, edge_index)
        frame_i = (x[row].unsqueeze(1) @ F_ij).squeeze(1)
        frame_j = (x[col].unsqueeze(1) @ F_ij).squeeze(1)
        frame_feats = torch.cat([frame_i, frame_j], dim=-1)
        force_feats = torch.cat([node_energy_hidden[row], node_energy_hidden[col], frame_feats], dim=-1)

        inv_force = self.force_mlp(force_feats)
        equiv_force = torch.sum(inv_force.unsqueeze(1) * F_ij, dim=-1)
        equiv_force = torch.clamp(equiv_force, min=-1e3, max=1e3)
        node_force = unsorted_segment_sum(equiv_force, row, num_segments=x.size(0),
                                   aggregation_method='mean')
        return node_force


class Frame_EGNN(nn.Module):
    def __init__(self, in_node_dim, in_bond_dim, hidden_dim, node_emb_dim=8, bond_emb_dim=4,
                 out_energy_dim=1, out_force_dim=3, device='cpu', batch_size=100, n_nodes=24,
                 act_fn=nn.SiLU(), n_layers=3, attention=True, 
                 num_rbf=16, rbf_trainable=False, aggregation_method='sum'):
        super(Frame_EGNN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.aggregation_method = aggregation_method
        self.rbf = GaussianSmearing(num_rbf=num_rbf, rbf_trainable=rbf_trainable)
        self.dis_dim = num_rbf + 1

        self.node_emb = nn.Linear(in_node_dim, node_emb_dim, bias=False)
        self.bond_emb = nn.Linear(in_bond_dim, bond_emb_dim, bias=False)

        self.add_module("gcl_0", GCL(
            node_emb_dim, bond_emb_dim, hidden_dim, hidden_dim, hidden_dim, self.dis_dim,
            aggregation_method=aggregation_method, 
            act_fn=act_fn, attention=attention))
        for i in range(1, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, self.dis_dim,
                aggregation_method=aggregation_method, 
                act_fn=act_fn, attention=attention))
        self.energy_block = energy_block(hidden_dim, hidden_dim, hidden_dim, self.dis_dim, batch_size,
                                         n_nodes, out_energy_dim, aggregation_method, act_fn)
        self.force_block = force_block(hidden_dim, out_force_dim, aggregation_method, act_fn)
        self.to(self.device)

    def forward(self, x, h, edge_index, bond):
        distances, _ = coord2diff(x, edge_index)
        rbf_emb = self.rbf(distances)
        dis_emb = torch.cat([distances, rbf_emb], dim=-1)

        h = self.node_emb(h)
        bond = self.bond_emb(bond)
        for i in range(0, self.n_layers):
            h, bond = self._modules["gcl_%d" % i](dis_emb, h, edge_index, bond)

        total_energy, node_energy_hidden =  self.energy_block(x, h, edge_index, bond, dis_emb)
        node_force = self.force_block(x, node_energy_hidden, edge_index)
        return total_energy, node_force

def localize(x, edge_index, norm_x_diff=True, edge_mask=None):
        row, col = edge_index

        if edge_mask is not None:
            x_diff = (x[row] - x[col]) * edge_mask
            x_cross = torch.cross(x[row], x[col]) * edge_mask
        else:
            x_diff = x[row] - x[col]
            x_cross = torch.cross(x[row], x[col])

        x_diff.to(device=x.device)
        x_cross.to(device=x.device)

        if norm_x_diff:
            # derive and apply normalization factor for `x_diff`
            if edge_mask is not None:
                norm = torch.ones((row.shape[-1], 1), device=x.device) * (1 - edge_mask)
                norm += torch.sqrt(torch.sum(((x_diff * edge_mask) ** 2  + 1e-30), dim=1, keepdim=True))
            else:
                norm = torch.sqrt(torch.sum(x_diff ** 2 + 1e-30, dim=1, keepdim=True)).to(device=x.device)
            x_diff = x_diff / norm

            # derive and apply normalization factor for `x_cross`
            if edge_mask is not None:
                cross_norm = torch.ones((row.shape[-1], 1), device=x.device) * (1 - edge_mask)
                cross_norm += torch.sqrt(torch.sum((x_cross * edge_mask) ** 2  + 1e-30, dim=1, keepdim=True))
                
            else:
                cross_norm = torch.sqrt(torch.sum(x_cross ** 2 + 1e-30, dim=1, keepdim=True)).to(device=x.device)
            x_cross = x_cross / cross_norm

        if edge_mask is not None:
            x_vertical = torch.cross(x_diff * edge_mask, x_cross * edge_mask)
        else:
            x_vertical = torch.cross(x_diff, x_cross)
        x_vertical.to(device=x.device)

        F_ij = torch.cat((x_diff.unsqueeze(-1), x_cross.unsqueeze(-1), x_vertical.unsqueeze(-1)), dim=-1)
        return F_ij

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1, keepdim=True)
    radial = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (radial + norm_constant)
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result