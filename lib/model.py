from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_add_pool
import torch
import torch.nn.functional as F
import torch.nn as nn

class GCNLayer_pyg(nn.Module):

    def __init__(self, in_feats, out_feats, gnn_norm='False', activation=None ,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer_pyg, self).__init__()

        self.activation = activation
        self.graph_conv = GCNConv(in_channels=in_feats, out_channels=out_feats,
                                    normalize=gnn_norm)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, x, edge_index):
        new_feats = self.graph_conv(x, edge_index)
        
        if self.residual:
            res_feats = self.activation(self.res_connection(x))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)
        return new_feats

class GCN_pyg(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__
    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    """

    def __init__(self, in_feats, hidden_feats=None, gnn_norm="False", activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCN_pyg, self).__init__()

        if hidden_feats is None:
            hidden_feats = [128, 128 ,128]

        n_layers = len(hidden_feats)
        if gnn_norm == "False":
            gnn_norm = ['False' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(gnn_norm), 
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer_pyg(in_feats, hidden_feats[i], gnn_norm[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

        #gnn_out_feats = self.hidden_feats[-1]
        #self.readout = WeightedSumAndMax(gnn_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    # def forward(self, g, feats):
    #     """Update node representations.
    #
    #     Parameters
    #     ----------
    #     g : DGLGraph
    #         DGLGraph for a batch of graphs
    #     feats : FloatTensor of shape (N, M1)
    #         * N is the total number of nodes in the batch of graphs
    #         * M1 is the input node feature size, which equals in_feats in initialization
    #
    #     Returns
    #     -------
    #     feats : FloatTensor of shape (N, M2)
    #         * N is the total number of nodes in the batch of graphs
    #         * M2 is the output node representation size, which equals
    #           hidden_sizes[-1] in initialization.
    #     """
    #     for gnn in self.gnn_layers:
    #         feats = gnn(g, feats)
    #     return feats

    def forward(self, x , edge_index, batch):
        #input = input
        # input = dgl.add_self_loop(input)
        #node_feats = input.ndata["h"]

        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
        # node_feats = super().forward(input, feats=node_feats)
        #graph_feats = self.readout(input, node_feats)
        graph_feats = global_add_pool(x,batch)
        return graph_feats


class Classify(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims_lst):
        '''
            input_dim (int)
            output_dim (int)
            hidden_dims_lst (list, each element is a integer, indicating the hidden size)
        '''
        super(Classify, self).__init__()
        layer_size = len(hidden_dims_lst) + 1
        dims = [input_dim] + hidden_dims_lst + [output_dim]

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(dims[i+1]) for i in range(layer_size)])
    def forward(self, v):
        # predict
        for i, l in enumerate(self.predictor):
            if i == len(self.predictor)-1:
                v = l(v)
            else:
                v=l(v)
                v= self.bn_layer[i](v)
                v = F.relu(v)
        return v


class Model(nn.Module):
    def __init__(self, gcn_hidden_dim, classify_input_dim, classify_hidden_dim, num_classes):
        super(Model, self).__init__()
        self.gcn_hidden_dim = gcn_hidden_dim
        self.classify_input_dim = classify_input_dim
        self.classify_hidden_dim = classify_hidden_dim
        self.num_classes = num_classes
        self.model_drug = GCN_pyg(in_feats = 74, hidden_feats = self.gcn_hidden_dim)
        self.classify = Classify(input_dim = self.classify_input_dim, output_dim = self.num_classes, hidden_dims_lst = self.classify_hidden_dim)

    def forward(self, x, graph, batch):
        x = self.model_drug(x, graph, batch)
        x = self.classify(x)
        return x


class SupConGCN(nn.Module):
    def __init__(self):
        super(SupConGCN,self).__init__()
        self.encoder = GCN_pyg(in_feats=74)
    def forward(self,x,edge_index,batch):
        feat=self.encoder(x,edge_index,batch)
        feat= F.normalize(feat, dim=1)
        return feat