import torch
from torch_geometric.graphgym import GATConv
import torch.nn.functional as F #To use an activation function or dropout
from torch.nn import MultiheadAttention, Conv1d, BatchNorm1d, MaxPool1d, Conv2d, BatchNorm2d, MaxPool2d, AvgPool1d
from torch.nn import Linear, ModuleList #The reason for defining the module list is to enable learning
from torch_geometric.nn import GATConv, TransformerConv, SAGPooling, TopKPooling, GraphNorm, BatchNorm
from torch_geometric.nn import global_mean_pool as gmp


class GraphMHC(torch.nn.Module):  # Inherited from nn.Module
    def __init__(self, hyperparameters):
        super(GraphMHC, self).__init__()
        in_channels = hyperparameters["in_channels"]
        channels = hyperparameters["channels"]
        heads = hyperparameters["heads"]
        self.heads = heads
        dropout_rate = hyperparameters["dropout_rate"]
        self.dropout_rate = dropout_rate
        edge_dim = hyperparameters["edge_dim"]
        kernel_size = hyperparameters["kernel_size"]

        self.conv1 = GATConv(in_channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False)
        self.norm1 = GraphNorm(channels)

        self.conv2 = GATConv(channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False)
        self.norm2 = GraphNorm(channels)

        self.conv3 = GATConv(channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False)
        self.norm3 = GraphNorm(channels)

        self.conv4 = GATConv(channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False)
        self.norm4 = GraphNorm(channels)

        self.conv5 = Conv1d(heads, heads * 2, kernel_size=kernel_size, padding='same')
        self.norm5 = BatchNorm1d(heads * 2)
        self.pool5 = AvgPool1d(2)

        self.conv6 = Conv1d(heads * 2, heads * 4, kernel_size=kernel_size, padding='same')
        self.norm6 = BatchNorm1d(heads * 4)
        self.pool6 = AvgPool1d(2)

        self.lin7 = Linear(channels, 1)

    def forward(self, x, edge_attr, edge_index, edge_weight, batch_index):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.norm4(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)

        x = gmp(x, batch_index)

        skip4 = x
        x = torch.reshape(x, (
        x.shape[0], self.heads, x.shape[1] // self.heads))  # put in as many channels as the number of heads

        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.pool5(x)
        skip5 = x.flatten(start_dim=1, end_dim=2)

        x = self.conv6(x)
        x = self.norm6(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.pool6(x)

        x = x.flatten(start_dim=1, end_dim=2)
        x += skip4 + skip5  # skip connection

        x = self.lin7(x)

        return x