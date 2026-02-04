import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

import ipdb
from model.drug_model.drug_encoder import drug_hier_encoder
from model.drug_model.GNNConv import *
from model.drug_model.GNNs import *
from model.drp_model.hetero_graph import HeteroGNN

from model.drp_model.self_attn import BinaryArrayEmbedding
import torch.nn.functional as F

class CLMT(torch.nn.Module):

    def __init__(
        self, global_para,
        omic_para,
        bert_model_name, 
        device, model_config
    ):
        super(CLMT, self).__init__()
        self.device = device

        self.alpha = 0.0
        self.alpha_step = 0.001
        self.max_alpha = 1.0


        # self.pathway_dim = 100
        # self.embed_dim = 64

        self.use_regulizer_cell = model_config.get('use_regulizer_cell')
        self.use_regulizer_drug = model_config.get('use_regulizer_drug')
        self.use_regulizer_pathway = model_config.get('use_drug_path_way')
        
        self.graph_embedding_dim = model_config.get('graph_embedding_dim')
        self.drug_dim = model_config.get('drug_embed_dim')
        self.dropout = model_config.get('dropout_rate')

        self.GNN_drug = drug_hier_encoder(global_para)


        # self.cell_protein_embedding = nn.Parameter(cell_protein_embedding)
        self.BatchNorm = nn.BatchNorm1d(320)

        self.heterograph = HeteroGNN(self.graph_embedding_dim) # 64

        self.drug_desc_path = model_config.get('drug_desc_path')
        self.cell_desc_path = model_config.get('cell_desc_path')
        
        self.drug_descriptions = pd.read_csv(self.drug_desc_path, index_col=0, encoding="latin-1")
        self.cell_descriptions = pd.read_json(self.cell_desc_path, encoding="utf-8")
        
        self.drug_id_to_desc = dict(zip(self.drug_descriptions['CID'].tolist(), self.drug_descriptions['Description'].tolist()))
        
        self.cell_id_to_desc = dict(zip(self.cell_descriptions['Cell_Line_ID'].tolist(), self.cell_descriptions['Text_Description'].tolist()))
    
        self.text_max_length = 512

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.text_extractor = AutoModel.from_pretrained(bert_model_name).to(device)

        bert_module = self.text_extractor
        drug_hier_encoder
        for param in bert_module.parameters():
                param.requires_grad = False
        
        # for param in bert_module.encoder.layer[11].parameters():
        #     param.requires_grad = True
        
        for param in bert_module.pooler.parameters():
            param.requires_grad = True

        # encode_text(texts, tokenizer, text_extractor, device)

        # ipdb.set_trace()

        self.cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(self.cell_descriptions['Cell_Line_ID'].tolist())}
        self.drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(self.drug_descriptions['CID'].tolist())}

        # self.cell_texts_embeddings = nn.Parameter(torch.cat([self.encode_text(self.cell_id_to_desc.get(cell_id, ""), self.tokenizer, self.text_extractor, self.text_max_length, device)
        #                    for cell_id in list(self.cell_id_to_desc.keys())]), requires_grad=True)
        # self.drug_texts_embeddings = nn.Parameter(torch.cat([self.encode_text(self.drug_id_to_desc.get(drug_id, ""), self.tokenizer, self.text_extractor, self.text_max_length, device)
        #                    for drug_id in list(self.drug_id_to_desc.keys())]), requires_grad=True)

        with torch.no_grad():

            all_cell_texts = [self.cell_id_to_desc.get(cid, "")[800:]
                              for cid in self.cell_descriptions['Cell_Line_ID'].tolist()]
            all_drug_texts = [self.drug_id_to_desc.get(did, "")[800:]
                              for did in self.drug_descriptions['CID'].tolist()]
            
            cell_text_embeds = self.encode_text(all_cell_texts, self.tokenizer, 
                                        self.text_extractor, 
                                        self.text_max_length, device)
            drug_text_embeds = self.encode_text(all_drug_texts, self.tokenizer,
                                        self.text_extractor,
                                        self.text_max_length, device)
        
        self.register_buffer('cell_text_cache', cell_text_embeds)
        self.register_buffer('drug_text_cache', drug_text_embeds)


        self.predictor = nn.Sequential(
            
            # nn.Linear(256*3+512, 512),
            # nn.Linear(1152, 512), 
            # nn.Linear(512+128*4, 512), 
            # nn.Linear(1280, 512),
            
            # nn.Linear(1280, 512),
            # nn.BatchNorm1d(512),  
            # nn.ELU(),

            nn.Linear(512+256*2+256*4, 512), # 512+256
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            # nn.Linear(512, 1)

            nn.Linear(512, 128), # 512+256
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 1)
            
        )

        self.cell_text_linear = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.drug_text_linear = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        # self.tower_linear = nn.Linear(512*4+64*4, 128)
        self.tower_linear = nn.Linear(512*3+64*4, 128)

        self.gene_linear = nn.Sequential(
            nn.Linear(omic_para['gen_dim'], 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        self.mut_linear = nn.Sequential(
            nn.Linear(omic_para['mut_dim'], 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )


        self.cnv_linear = nn.Sequential(
            nn.Linear(omic_para['cnv_dim'], 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )


        self.node_drug_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        # self.mut_selfA = BinaryArrayEmbedding(omic_para['mut_dim'])
        # self.cnv_selfA = BinaryArrayEmbedding(omic_para['cnv_dim'])


        if self.use_regulizer_cell == 'True':
            # self.cell_regulizer = nn.Linear(256 * 6, 26) ## 26 cancer types
            self.cell_regulizer = nn.Linear(512, omic_para['cell_dim']) ## 26 cancer types 128

        if self.use_regulizer_drug == 'True':
            # self.drug_regulizer = nn.Sequential(nn.Linear(512 , 1024), 
            self.drug_regulizer = nn.Sequential(nn.Linear(512 , 8), 
                                            nn.ReLU(),
                                            nn.Linear(8,1))
        if self.use_regulizer_pathway == 'True':
            # self.drug_path_way_class = nn.Sequential(nn.Linear(512 , 1024), 
            self.drug_path_way_class = nn.Sequential(nn.Linear(512 , 32), 

                                            nn.ReLU(),
                                            nn.Linear(32,omic_para['drug_pathway_dim']))

        # self.linear1 = nn.Linear(64, 64)drug_pathway
        # self.linear2 = nn.Linear(128, 64)


    def encode_text(self, texts, tokenizer, text_extractor, text_max_length, device):
        

        def meanpooling(output, mask):
            embeddings = output[0] # First element of model_output contains all token embeddings
            mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
            return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=text_max_length)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        # Compute token embeddings
        output = text_extractor(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        embeddings = meanpooling(output, encoded_input['attention_mask'])
        
        # print("Sentence embeddings:")
        # print(embeddings.size())
        
        return embeddings


    def forward(self, batch,
                drug_atom, drug_bond,  
                gen_feats, mut_feats, cnv_feats,
                batch_cell_name, batch_drug_name,
                device):
        


        cell_expression = batch['cell_line'].x.to(device)

        x_dict = {

            'cell_line': cell_expression,
            'drug': batch['drug'].x.to(device),
        }

        edge_index_dict = {

            ('cell_line', 'response', 'drug'): batch['cell_line', 'response', 'drug'].edge_index.to(device),
            ('cell_line', 'similar', 'cell_line'): batch['cell_line', 'similar', 'cell_line'].edge_index.to(device),
            ('drug', 'similar', 'drug'): batch['drug', 'similar', 'drug'].edge_index.to(device),
        }

        batch_edge_index = batch['cell_line', 'response', 'drug'].edge_label_index

        out_dict = self.heterograph(x_dict, edge_index_dict)

        cell_line_embeddings = out_dict['cell_line']
        drug_embeddings = out_dict['drug']

        cell_graph_embedding = cell_line_embeddings[batch_edge_index[0]]
        drug_graph_embedding = drug_embeddings[batch_edge_index[1]]


        # # 'Dual-tower Embedding'
        x_drug = self.GNN_drug(drug_atom, drug_bond) # [512] 
        x_exp = self.gene_linear(gen_feats) # 256
        
        x_mut = self.mut_linear(mut_feats) # [128]
        x_cnv = self.cnv_linear(cnv_feats) # [128]

        # # 'Text Embedding'

        batch_cell_indices = [self.cell_id_to_idx[i] for i in batch_cell_name]
        batch_drug_indices = [self.drug_id_to_idx[i] for i in batch_drug_name]


        cell_text_embedding = self.cell_text_linear(self.cell_text_cache[batch_cell_indices])
        drug_text_embedding = self.drug_text_linear(self.drug_text_cache[batch_drug_indices])

    
        graph_view = torch.cat([cell_graph_embedding, drug_graph_embedding], dim=1) # [batch,128+128]

        # # cell_view= self.tower_linear(torch.cat([x_dg, x_dm, x_dc], dim=1))

        # # cell_view  = torch.cat([x_mut], dim=1) # 256+512
        # # cell_view  = torch.cat([x_exp, x_cnv], dim=1) # 256+512
        
        cell_view  = torch.cat([x_exp, x_mut, x_cnv], dim=1) # 256+512
        # # cell_view  = torch.cat([x_drug], dim=1) # 256+512

        tower_view  = torch.cat([cell_view, x_drug], dim=1) # 256+512

        text_view  = torch.cat([cell_text_embedding, drug_text_embedding], dim=1) #[batch,768->128+128]

        # views = [text_view] # 256*4+512*2

        # views = [x_drug, cell_view] # 256

        # print(x_drug.size(), cell_view.size())

        views = [text_view, graph_view, tower_view] #[512, 256, 256]

        combined = torch.cat(views, dim=1)

        prediction = self.predictor(combined)


        cell_class = None
        drug_class = None
        drug_pathway = None

        if self.use_regulizer_cell == 'True':
            cell_class = self.cell_regulizer(cell_view)
        if self.use_regulizer_drug == 'True':
            drug_class = self.drug_regulizer(x_drug)
        if self.use_regulizer_pathway =='True':
            drug_pathway = self.drug_path_way_class(x_drug)

        
        return {
            'pred': prediction,
            'cell_regulizer':cell_class, 'drug_regulizer': drug_class, 'drug_pathway':drug_pathway,
            'features' : [x_exp, self.node_drug_linear(x_drug), cell_graph_embedding, drug_graph_embedding, cell_text_embedding, drug_text_embedding],
            }
    