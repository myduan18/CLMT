from torch_geometric.nn import SAGEConv, HeteroConv
import torch
import torch.nn.functional as F

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(HeteroGNN, self).__init__()

        self.conv1 = HeteroConv({
            ('cell_line', 'response', 'drug'): SAGEConv(-1, hidden_channels), 
            ('cell_line', 'similar', 'cell_line'): SAGEConv(-1, hidden_channels),
            ('drug', 'similar', 'drug'): SAGEConv(-1, hidden_channels),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('cell_line', 'response', 'drug'): SAGEConv(hidden_channels, hidden_channels),
            ('cell_line', 'similar', 'cell_line'): SAGEConv(hidden_channels, hidden_channels),
            ('drug', 'similar', 'drug'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()} 

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict
    


