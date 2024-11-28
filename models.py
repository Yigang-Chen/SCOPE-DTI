import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_batch
from gvp import GVP, GVPConvLayer, LayerNorm
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from dgl.nn.pytorch.conv import GraphConv, GATConv
from dgl.nn.pytorch import HeteroGraphConv
import functools

class DrugGVPModel(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, max_node=300, num_layers=3, drop_rate=0.1):
        """
        Parameters are extracted from a configuration dictionary
        to make the model more flexible and modular.
        """
        super(DrugGVPModel, self).__init__()

        # Node transformation layers
        self.node_layer_norm = LayerNorm(node_in_dim)
        self.node_gvp = GVP(node_in_dim, node_h_dim, activations=(None, None))
        
        # Edge transformation layers
        self.edge_layer_norm = LayerNorm(edge_in_dim)
        self.edge_gvp = GVP(edge_in_dim, edge_h_dim, activations=(None, None))

        # GVPConv layers with configurable number of layers
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        # Output layer for node features
        ns, _ = node_h_dim
        self.output_layer_norm = LayerNorm(node_h_dim)
        self.output_gvp = GVP(node_h_dim, (ns, 0))
        self.max_node_len = max_node

    def forward(self, xd):
        """
        Forward pass of the model.
        """
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)  # Node scalar and vector features
        h_E = (xd.edge_s, xd.edge_v)  # Edge scalar and vector features
        edge_index = xd.edge_index    # Edge index for message passing
        batch = xd.batch              # Batch index for pooling

        # Apply node transformation (layer norm + GVP)
        h_V = self.node_layer_norm(h_V)
        h_V = self.node_gvp(h_V)
        
        # Apply edge transformation (layer norm + GVP)
        h_E = self.edge_layer_norm(h_E)
        h_E = self.edge_gvp(h_E)
        
        # Apply GVPConv layers
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Apply output transformation (layer norm + GVP)
        h_V = self.output_layer_norm(h_V)
        out = self.output_gvp(h_V)

        # Perform global pooling over the batch
        out1 = torch_geometric.nn.global_add_pool(out, batch)

        return out1

# original version
class ProteinGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, prot_len=2000, padding=True, fc_bias=True, drop_rate=0.2):
        assert padding
        super(ProteinGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.prot_len = prot_len
        self.fc_bias = fc_bias

        cascade = [in_dim + i * (out_dim - in_dim) // num_layers for i in range(num_layers)] + [out_dim]
        # print("cascade", cascade)

        for i in range(num_layers):
            self.convs.append(HeteroGraphConv({
                "knn_edge": GraphConv(cascade[i], cascade[i + 1]),
                "r_sphere_edge": GraphConv(cascade[i], cascade[i + 1]),
                "seq_edge": GraphConv(cascade[i], cascade[i + 1]),
            }, aggregate="sum"))
            self.bns.append(nn.BatchNorm1d(cascade[i + 1]))
            self.fcs.append(nn.Linear(cascade[i + 1], cascade[i + 1], bias=self.fc_bias))

    def forward(self, g):
        h = g.ndata['x']

        for i in range(self.num_layers):
            h = self.convs[i](g, {"residue": h})
            h = self.bns[i](h['residue'])
            h = torch.relu(h)
            h = self.fcs[i](h)

        h = h.view(-1, self.prot_len, self.out_dim)

        return h  

class Scope(nn.Module):
    def __init__(self, **config):
        super(Scope, self).__init__()
        drug_node_in_dim = config["DRUG"]["ATOM_IN_DIM"]
        drug_node_h_dim = config["DRUG"]["ATOM_HIDDEN_DIM"]
        drug_edge_in_dim = config["DRUG"]["EDGE_IN_DIM"]
        drug_edge_h_dim = config["DRUG"]["EDGE_HIDDEN_DIM"]
        drug_num_layers = config["DRUG"]["NUM_LAYERS"]
        drug_drop_rate = config["DRUG"]["DROP_RATE"]
        drug_max_node = config["DRUG"]["MAX_NODES"]

        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_padding = config["PROTEIN"]["PADDING"]
        protein_max_length = config["PROTEIN"]["MAX_LENGTH"] #new
        protein_num_layers = config["PROTEIN"]["GRAPH"]["NUM_LAYER"]
        protein_fc_bias = config["PROTEIN"]["GRAPH"]["FC_BIAS"]

        ban_heads = config["BCN"]["HEADS"]

        self.drug_extractor = DrugGVPModel(node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dim,
                                           edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dim,
                                           num_layers=drug_num_layers, drop_rate=drug_drop_rate, max_node=drug_max_node)

        self.protein_featurizer = ProteinGCN(7, protein_emb_dim, num_layers=protein_num_layers, prot_len=protein_max_length, fc_bias=protein_fc_bias)

        assert drug_node_h_dim[0] == protein_emb_dim
        q_dim = protein_emb_dim

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_node_h_dim[0], q_dim=q_dim, h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None,
        )

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, xd, xp):

        v_d = self.drug_extractor(xd)
        v_d = v_d.unsqueeze(1)

        v_p = self.protein_featurizer(xp)
        
        f, att = self.bcn(v_d, v_p)

        score = self.mlp_classifier(f)
        return v_d, v_p, f, score

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)  # Ensure out_dim is 128 (or expected feature dimension)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape  # (b, 2000, 1280)
        x = x.view(-1, feature_dim)  # Flatten to (b * 2000, 1280) for Linear
        x = F.relu(self.fc1(x))      # Apply Linear: (b * 2000, 128)
        x = self.bn1(x)              # Apply BatchNorm1d on the last dimension
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (b, 2000, 128)

        return x

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

