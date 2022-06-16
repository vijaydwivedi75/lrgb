import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from performer_pytorch import SelfAttention


class GPSwLPELayer(nn.Module):
    """ Local MPNN + full graph attention x-former layer
        with learnable positional encodings.

    WARNING: This is an abandoned experimental code, to be deleted.
    """

    def __init__(self, dim_h, dim_pe,
                 local_gnn_type, global_model_type,
                 num_heads, full_graph,
                 pna_degrees=None, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super().__init__()

        self.dim_h = dim_h
        self.dim_pe = dim_pe
        self.dim_hcrop = dim_h - dim_pe
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                              out_channels=dim_h // num_heads,
                                              heads=num_heads,
                                              edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                              aggregators=aggregators,
                                              scalers=scalers,
                                              deg=deg,
                                              edge_dim=dim_h,
                                              towers=1,
                                              pre_layers=1,
                                              post_layers=1,
                                              divide_input=False)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        dim_mult = int(self.local_model is not None) + \
                   int(self.self_attn is not None)
        self.O_h = nn.Sequential(nn.Linear(dim_h * dim_mult, dim_h * dim_mult),
                                 nn.ReLU(),
                                 nn.Linear(dim_h * dim_mult, self.dim_hcrop))

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(self.dim_hcrop)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(self.dim_hcrop)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(self.dim_hcrop, self.dim_hcrop * 2)
        self.FFN_h_layer2 = nn.Linear(self.dim_hcrop * 2, self.dim_hcrop)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(self.dim_hcrop)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(self.dim_hcrop)

        # Separate model for PE.
        pe_gin_nn = nn.Sequential(Linear_pyg(dim_pe, dim_pe),
                                  nn.ReLU(),
                                  Linear_pyg(dim_pe, dim_pe))
        self.pe_model = pygnn.GINEConv(pe_gin_nn, edge_dim=dim_h)
        if self.layer_norm:
            self.layer_norm_pe = nn.LayerNorm(self.dim_pe)
        if self.batch_norm:
            self.batch_norm_pe = nn.BatchNorm1d(self.dim_pe)


    def forward(self, batch):
        h = batch.x
        h_in1 = h[..., :self.dim_hcrop]  # For first residual connection.
        pe_in = h[..., self.dim_hcrop:]  # Split out PE for separate processing.

        h_out_list = []
        # Local MPGNN with edge attributes.
        if self.local_model is not None:
            h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
            h_local = F.relu(h_local)
            h_out_list.append(h_local)
        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn_out = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn_out = self.self_attn(h_dense, mask=mask)[mask]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")
            h_out_list.append(h_attn_out)

        # Concat local and global outputs
        h = torch.cat(h_out_list, dim=-1)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        # Update PE.
        pe = self.pe_model(pe_in, batch.edge_index, batch.edge_attr)
        pe = F.relu(pe)
        if self.layer_norm:
            pe = self.layer_norm_pe(pe)
        if self.batch_norm:
            pe = self.batch_norm_pe(pe)
        if self.residual:
            pe = pe_in + pe

        # Concatenate updated PE to node embeddings.
        batch.x = torch.cat((h, pe), 1)
        return batch

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask) :
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def __repr__(self):
        return '{}(dim_h={}, local_gnn_type={}, global_model_type={}, heads={}, residual={})'.format(
            self.__class__.__name__, self.dim_h, self.local_gnn_type,
            self.global_model_type, self.num_heads, self.residual)
