import math

import torch
import torch.nn.functional as func
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class GRUmp(MessagePassing):
    def __init__(self,
                 out_channels,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super(GRUmp, self).__init__(aggr=aggr, **kwargs)
        self.out_channels = out_channels
        self.bias = Parameter(torch.Tensor(out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.bias)
        self.rnn.reset_parameters()

    def forward(self, h, edge_index, edge_weight):
        m = torch.add(h, torch.stack([self.bias] * h.size(0)))
        m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
        h = self.rnn(m, h)
        return h

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, num_msg={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)

class QGNN(torch.nn.Module):
    def __init__(self,
                 out_channels=160,
                 num_msg=20,
                 dropout_ratio=.15,
                 packet_loss=.15,
                 **kwargs):
        super(QGNN, self).__init__()
        self.out_channels = out_channels
        self.num_msg = num_msg 
        self.dropout_ratio = dropout_ratio
        self.packet_loss = packet_loss

        self.dmp = GRUmp(out_channels, **kwargs)
        self.dmp_norm = torch.nn.Sequential(
            torch.nn.LayerNorm([out_channels]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.dropout_ratio))

        self.linA = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.LayerNorm([out_channels]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.dropout_ratio))
        self.linB = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.LayerNorm([out_channels]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.dropout_ratio))

        self.Q = torch.nn.Sequential(
            torch.nn.Linear(32, out_channels),
            torch.nn.LayerNorm([out_channels]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.dropout_ratio))
        self.F = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 1),
            torch.nn.Sigmoid())

    def zero_right_pad(self, x):
        if x.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')
        if x.size(1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(1))
            x = torch.cat([x, zero], dim=1)
        return x

    def attention(self, x, edge_index):
        att_weight = torch.sigmoid(torch.matmul(self.linA(x), self.linB(x).t()))
        edge_weight = att_weight[edge_index[0], edge_index[1]]
        if self.training:
            mask = torch.bernoulli(torch.Tensor([1 - self.packet_loss] * edge_weight.size(0)))
            edge_weight = mask * edge_weight
        return edge_weight

    def routing(self, x, num_interfaces, num_routers, targets):
        cum_num_nodes = torch.cumsum(torch.add(num_routers, num_interfaces), dim=0)
        interfaces_x = torch.cat([x[(n - num_interfaces[i]):n] for i, n in enumerate(cum_num_nodes)])
        mult_targets = torch.cat([torch.stack([targets[i]] * n) for i, n in enumerate(num_interfaces)])
        q = self.Q(mult_targets)
        o = self.F(q * interfaces_x)
        return o

    def forward(self, data):
        x, edge_index, num_routers, num_interfaces, targets = data.x, data.edge_index, data.num_routers, data.num_interfaces, data.targets

        h = x if x.dim() == 2 else x.unsqueeze(-1)
        h = self.zero_right_pad(h)
        edge_weight = torch.ones(edge_index.size(1))
        for _ in range(self.num_msg):
            h = self.dmp(h, edge_index, edge_weight)
            h = self.dmp_norm(h)
            edge_weight = self.attention(h, edge_index)
        o = self.routing(h, num_interfaces, num_routers, targets)
        return o
