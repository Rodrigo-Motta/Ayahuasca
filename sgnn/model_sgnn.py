import torch
import torch.nn as nn
import torch.nn.functional as func
from torch_geometric.nn import ChebConv, GCNConv, global_mean_pool, GATv2Conv, global_max_pool

class GCN(torch.nn.Module):
    """GCN model(network architecture can be modified)"""

    def __init__(self,
                 num_features,
                 k_order,
                 dropout=.6):
        super(GCN, self).__init__()

        self.p = dropout

        self.conv1 = ChebConv(int(num_features), 64, K=k_order)
        self.bn1 = torch.nn.BatchNorm1d(64)

        # self.gat1 = GATv2Conv(int(num_features), 32, heads=4)
        # self.bn1 = torch.nn.BatchNorm1d(32*4)

        self.conv2 = ChebConv(64, 32, K=k_order)
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.conv3 = ChebConv(32, 16, K=k_order)
        self.bn3 = torch.nn.BatchNorm1d(16)

        # self.conv4 = ChebConv(64, 32, K=k_order)
        # self.bn4 = torch.nn.BatchNorm1d(32)

        self.lin1 = torch.nn.Linear(16, 2)
        #self.lin2 = torch.nn.Linear(8, 2)

        #self.skip_connection = nn.Linear(num_features, 16)

        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    
        batch = data.batch

        # Transform the input features to match the output features dimension
        #skip_x = self.skip_connection(x)

        # x = torch.nn.functional.leaky_relu(
        #     self.bn1(
        #         self.gat1(x, edge_index)))  ##, edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        # x = func.dropout(x, p=self.p, training=self.training)
        

        x = torch.nn.functional.leaky_relu(
            self.bn1(
                self.conv1(x, edge_index)))  ##, edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        x = func.dropout(x, p=self.p, training=self.training)


        x = torch.nn.functional.leaky_relu(
            self.bn2(
                self.conv2(x, edge_index)))  # , edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        x = func.dropout(x, p=self.p, training=self.training)

        x = torch.nn.functional.leaky_relu(
            self.bn3(
                self.conv3(x, edge_index)))  # , edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        x = func.dropout(x, p=self.p, training=self.training)

        # x = torch.nn.functional.leaky_relu(
        #     self.bn4(
        #         self.conv4(x, edge_index)))  # , edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        # x = func.dropout(x, p=self.p, training=self.training)

        # Add skip connection
        #x += skip_x

        x = self.pool(x, batch)
        x = self.lin1(x)
        #x = torch.nn.functional.leaky_relu(self.lin1(x))
        # x = func.dropout(x, p=self.p, training=self.training)
       # x = self.lin2(x)

        return x

class SiameseNetwork(nn.Module):
    def __init__(self, num_features, k_order, dropout=.6):
        super().__init__()
        self.gcn = GCN(num_features, k_order, dropout)

    def forward_one(self, x):
        return self.gcn(x)

    def forward(self, input1, input2):
        output1 = self.gcn(input1)
        output2 = self.gcn(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 2) -> None: #margin =1 #margin=.5
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive