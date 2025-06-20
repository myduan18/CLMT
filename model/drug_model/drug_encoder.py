import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, JumpingKnowledge
from torch_geometric.nn.norm import GraphNorm 
import torch_geometric.nn as pyg_nn
import sys
sys.path.append('..')
from model.drug_model.drug_gnn import *

class drug_conv_1d(torch.nn.Module): ### 3 convolution layer + 2 linear transform
    def __init__(self, embed_dim,in_channels = 30,out_channels = [40,80,60],smiles_in_dim = 360):  ### in_channels is dim of one hot embedding for smiles. ## smiles_in_dim is dimension after 3 convolution steps.
        super(drug_conv_1d,self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv_smiles_1 = nn.Conv1d(in_channels= self.in_channels,out_channels =self.out_channels[0] ,kernel_size =7,stride=1,padding='same')
        self.smiles_bn1 = nn.BatchNorm1d(self.out_channels[0])
        self.smiles_pool1 = nn.MaxPool1d(3)
        self.smiles_act1 = nn.ReLU()
        self.smiles_conv_block_1 = nn.Sequential(self.conv_smiles_1,self.smiles_bn1,self.smiles_act1,self.smiles_pool1)
        self.conv_smiles_2 = nn.Conv1d(in_channels= out_channels[0],out_channels =self.out_channels[1] ,kernel_size =7,stride=1,padding='same')
        self.smiles_bn2 = nn.BatchNorm1d(out_channels[1])
        self.smiles_pool2 = nn.MaxPool1d(3)
        self.smiles_act2 = nn.ReLU()
        self.smiles_conv_block_2 = nn.Sequential(self.conv_smiles_2,self.smiles_bn2,self.smiles_act2,self.smiles_pool2)
        self.conv_smiles_3 = nn.Conv1d(in_channels= out_channels[1],out_channels =self.out_channels[2] ,kernel_size =7,stride=1,padding='same')
        self.smiles_bn3 = nn.BatchNorm1d(out_channels[2])
        self.smiles_pool3 = nn.MaxPool1d(3)
        self.smiles_act3 = nn.ReLU()
        self.smiles_conv_block_3 = nn.Sequential(self.conv_smiles_3,self.smiles_bn3,self.smiles_act3,self.smiles_pool3)
        self.ln_smiles_1 = nn.Linear(smiles_in_dim,512)
        self.bn_smiles = nn.BatchNorm1d(512)
        self.act_smiles1 = nn.ReLU()
        self.ln_smiles2 = nn.Linear(512,embed_dim)
        self.bn_smiles2 = nn.BatchNorm1d(embed_dim)
        self.embed_smiles = nn.Sequential(self.ln_smiles_1,self.bn_smiles,self.act_smiles1,self.ln_smiles2,self.bn_smiles2)
    def forward(self,smiles):
        out = self.smiles_conv_block_1(smiles.float())
        out = self.smiles_conv_block_2(out)
        out = self.smiles_conv_block_3(out)
        out = out.view(-1,out.shape[1]*out.shape[2])
        smiles_out = self.embed_smiles(out)
        return smiles_out 

class drug_fp_conv_1d(torch.nn.Module): ## 3 convolution layer + 2 linear transform
    def __init__(self, embed_dim, fp_in_dim = 1152,n_filters = 4): ## fp_in_dim is dimension after 3 convolution steps.
        super(drug_fp_conv_1d,self).__init__()
        self.conv_fp_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.fp_bn1 = nn.BatchNorm1d(n_filters)
        self.fp_pool1 = nn.MaxPool1d(3)
        self.fp_act1 = nn.ReLU()
        self.fp_conv_block_1 = nn.Sequential(self.conv_fp_1,self.fp_bn1,self.fp_act1,self.fp_pool1)
        self.conv_fp_2 = nn.Conv1d(in_channels= n_filters,out_channels =n_filters * 2 ,kernel_size =8)
        self.fp_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.fp_pool2 = nn.MaxPool1d(3)
        self.fp_act2 = nn.ReLU()
        self.fp_conv_block_2 = nn.Sequential(self.conv_fp_2,self.fp_bn2,self.fp_act2,self.fp_pool2)
        self.conv_fp_3 = nn.Conv1d(in_channels= n_filters * 2,out_channels =n_filters * 4 ,kernel_size =8)
        self.fp_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.fp_pool3 = nn.MaxPool1d(3)
        self.fp_act3 = nn.ReLU()
        self.fp_conv_block_3 = nn.Sequential(self.conv_fp_3,self.fp_bn3,self.fp_act3,self.fp_pool3)
        self.ln_fp_1 = nn.Linear(fp_in_dim,512)
        self.bn_fp = nn.BatchNorm1d(512)
        self.act_fp1 = nn.ReLU()
        self.ln_fp2 = nn.Linear(512,embed_dim)
        self.bn_fp2 = nn.BatchNorm1d(embed_dim)        
        self.embed_fp = nn.Sequential(self.ln_fp_1,self.bn_fp,self.act_fp1,self.ln_fp2,self.bn_fp2)
     
    def forward(self,smiles):
        out = self.fp_conv_block_1(smiles)
        out = self.fp_conv_block_2(out)
        out = self.fp_conv_block_3(out)
        out = out.view(-1,out.shape[1]*out.shape[2])
        fp_out = self.embed_fp(out)
        return fp_out 

class drug_fp_mlp(nn.Module):
    def __init__(self, embed_dim, input_features = 2048,dropout_rate = 0.4):
        super(drug_fp_mlp, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x.squeeze(1))


## Graph geom learning
class Drug_3d_Encoder(nn.Module):
    def __init__(self,model_config):
        super(Drug_3d_Encoder, self).__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.layer_num = model_config.get('layer_num')
        self.readout = model_config.get('readout')
        self.jk = model_config.get('JK')
        self.atom_init_nn = atom_embedding_net(model_config)
        self.bond_init_nn = bond_embedding_net(model_config)
        self.atom_conv = nn.ModuleList()
        self.bond_conv = nn.ModuleList()
        self.bond_embed_nn = nn.ModuleList()
        self.bond_angle_embed_nn = nn.ModuleList()
        self.batch_norm_atom = nn.ModuleList()
        self.layer_norm_atom = nn.ModuleList()
        self.graph_norm_atom = nn.ModuleList()
        self.batch_norm_bond = nn.ModuleList()
        self.layer_norm_bond = nn.ModuleList()
        self.graph_norm_bond = nn.ModuleList()
        self.JK = JumpingKnowledge('cat')
        for i in range(self.layer_num):
            self.atom_conv.append(pyg_nn.GINEConv(nn = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim*2), nn.ReLU(), nn.Linear(self.embed_dim*2, self.embed_dim)),edge_dim=self.embed_dim))
            self.batch_norm_atom.append(nn.BatchNorm1d(self.embed_dim))
            self.layer_norm_atom.append(nn.LayerNorm(self.embed_dim))
            self.graph_norm_atom.append(GraphNorm(self.embed_dim))
            self.bond_conv.append(pyg_nn.GINEConv(nn = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim*2), nn.ReLU(), nn.Linear(self.embed_dim*2, self.embed_dim)),edge_dim=self.embed_dim))
            self.bond_embed_nn.append(bond_embedding_net(model_config))
            self.bond_angle_embed_nn.append(nn.Sequential(nn.Linear(1,self.embed_dim),nn.ReLU(),nn.Linear(self.embed_dim,self.embed_dim)))
            self.batch_norm_bond.append(nn.BatchNorm1d(self.embed_dim))
            self.batch_norm_bond.append(nn.BatchNorm1d(self.embed_dim))
            self.layer_norm_bond.append(nn.LayerNorm(self.embed_dim))
            self.graph_norm_bond.append(GraphNorm(self.embed_dim))
        if self.readout == 'max':
            self.read_out = pyg_nn.global_max_pool
        elif self.readout == 'mean':
            self.read_out = pyg_nn.global_mean_pool
        elif self.readout == 'add':
            self.read_out = pyg_nn.global_mean_pool
    def forward(self,drug_atom, drug_bond):
        x, edge_index, edge_attr, batch = drug_atom.x, drug_atom.edge_index, drug_atom.edge_attr, drug_atom.batch
        x = self.atom_init_nn(x.to(dtype=torch.int64))
        edge_hidden = self.bond_init_nn(edge_attr.to(dtype=torch.int64))
        hidden = [x]
        hidden_edge = [edge_hidden]
        for i in range(self.layer_num):
            x = self.atom_conv[i](x = x, edge_attr = hidden_edge[i], edge_index = edge_index)
            x = self.layer_norm_atom[i](x)
            x = self.graph_norm_atom[i](x)
            if i == self.layer_num - 1:
                x = nn.Dropout(self.dropout_rate)(nn.ReLU()(x)) + hidden[i]
            else:
                x = nn.Dropout(self.dropout_rate)(x) + hidden[i]
            # x = self.batch_norm_atom[i](x)  
            cur_edge_attr = self.bond_embed_nn[i](edge_attr)    
            cur_angle_attr = self.bond_angle_embed_nn[i](drug_bond.edge_attr)     
            edge_hidden = self.bond_conv[i](x = cur_edge_attr,edge_attr = cur_angle_attr,edge_index = drug_bond.edge_index)
            # edge_hidden = self.batch_norm_bond[i](edge_hidden)
            edge_hidden = self.layer_norm_bond[i](edge_hidden)
            edge_hidden = self.graph_norm_bond[i](edge_hidden)
            if i == self.layer_num - 1:
                edge_hidden = nn.Dropout(self.dropout_rate)(nn.ReLU()(edge_hidden)) + hidden_edge[i]
            else:
                edge_hidden = nn.Dropout(self.dropout_rate)(edge_hidden) + hidden_edge[i]
            hidden.append(x)
            hidden_edge.append(edge_hidden)
        if self.jk == 'True':
            x = self.JK(hidden)
        else: x = hidden[-1]
        graph_repr = self.read_out(x, batch)
        # graph_repr = F.dropout(graph_repr, p=self.dropout_rate, training=self.training)   
        return graph_repr
    @property
    def output_dim(self):
        self.out_dim = self.embed_dim * (self.layer_num + 1)
        return self.out_dim


class drug_hier_encoder(nn.Module):
    def __init__(self,model_config) -> None:
        super().__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.drug_3d = Drug_3d_Encoder(model_config)
        self.drug_fp = drug_fp_conv_1d(embed_dim = self.embed_dim)
        # self.drug_fp = drug_fp_mlp(self.embed_dim)
        self.drug_3d_dim = self.drug_3d.output_dim
        self.drug_3d_dense = nn.Linear(self.drug_3d_dim, 1024)
        self.drug_3d_dense_bn1 = nn.BatchNorm1d(1024)
        self.drug_3d_dense2 = nn.Linear(1024, self.embed_dim)
        self.drug_3d_dense_bn2 = nn.BatchNorm1d(self.embed_dim)
    def forward(self,drug_atom,drug_bond):
        
        drug_repr = self.drug_3d(drug_atom,drug_bond)
        drug_repr = self.drug_3d_dense(drug_repr)
        drug_repr = self.drug_3d_dense_bn1(drug_repr)
        drug_repr = self.drug_3d_dense2(drug_repr)
        drug_repr = self.drug_3d_dense_bn2(drug_repr)

        import ipdb
        # ipdb.set_trace()
        batch_size = drug_repr.size(0)
        # batch_size = int(drug_atom.one_hot_smiles.shape[0]/30)
        fp = drug_atom.fp.view(batch_size,1,2048)

        # ipdb.set_trace()

        fp_repr = self.drug_fp(fp)


        return torch.cat([fp_repr,drug_repr], dim = -1) 
 
    
