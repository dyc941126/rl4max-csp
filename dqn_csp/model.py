import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=4, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(32, 8, heads=4, edge_dim=edge_dim, concat=True)
        self.conv3 = GATConv(32, 8, heads=4, edge_dim=edge_dim, concat=True)
        self.conv4 = GATConv(32, out_channels, heads=4, edge_dim=edge_dim, concat=False)
        self.pooling_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.target_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.out = nn.Linear(2 * out_channels, 1)

    def forward(self, batch, scatter_index, scatter_norm, decision_var_indexes):
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv3(x, batch.edge_index, batch.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv4(x, batch.edge_index, batch.edge_attr)
        x = F.leaky_relu(x)

        shifted_scatter_index = []
        shifted_decision_var_indexes = []
        s = 0
        nodes = []
        idx = 0
        acc_num_node = 0
        for i in range(batch.num_graphs):
            num_nodes = scatter_index[i][-1].item() + 1
            nodes += [idx] * num_nodes
            idx += 1
            shifted_decision_var_indexes.append(decision_var_indexes[i] + s)
            shifted_scatter_index.append(scatter_index[i] + acc_num_node)
            g = batch.get_example(i)
            s += g.x.shape[0]
            acc_num_node += num_nodes
        shifted_scatter_index = torch.cat(shifted_scatter_index, dim=0)
        nodes = torch.tensor(nodes, dtype=torch.long, device=x.device)

        pooling = scatter_sum(x, shifted_scatter_index, dim=0)
        assert pooling.shape[0] == scatter_norm.shape[0]
        pooling = pooling / scatter_norm
        pooling = scatter_sum(pooling, nodes, dim=0)
        target = x[shifted_decision_var_indexes]

        pooling = self.pooling_proj(pooling)
        target = self.target_proj(target)

        return self.out(F.leaky_relu(torch.cat([pooling, target], dim=1)))

    @torch.no_grad()
    def inference(self, data, scatter_index, scatter_norm, decision_var_indexes):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv3(x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv4(x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)

        pooling = scatter_sum(x, scatter_index, dim=0)
        assert pooling.shape[0] == scatter_norm.shape[0]
        pooling = pooling / scatter_norm
        pooling = pooling.sum(0, keepdim=True)
        pooling = pooling.repeat(len(decision_var_indexes), 1)
        target = x[decision_var_indexes]

        pooling = self.pooling_proj(pooling)
        target = self.target_proj(target)

        return self.out(F.leaky_relu(torch.cat([pooling, target], dim=1)))