import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_lpe_layer import GPSwLPELayer


class GPSwLPEModel(torch.nn.Module):
    """ GPS x-former with learnable positional encodings.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        # Sum up `dim_pe` of all PE encoders that are enabled.
        total_pe_dim = 0
        for k, subcfg in cfg.items():
            if k.startswith('posenc_') and subcfg.enable:
                total_pe_dim += subcfg.dim_pe
        # Parse local MPNN and xformer layer types.
        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSwLPELayer(
                dim_h=cfg.gt.dim_hidden,
                dim_pe=total_pe_dim,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                full_graph=cfg.gt.full_graph,  # Not used ATM
                pna_degrees=cfg.gt.pna_degrees,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=cfg.gt.residual))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


register_network('GPSwLPEModel', GPSwLPEModel)
register_network('GPSModel+LPE', GPSwLPEModel)