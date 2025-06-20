import sys
sys.path.append('prepare_data')
from create_cell_feat import *
from create_drug_feat import *
from create_drp_dict import *
from binarization_drp import *
from torch_geometric.data import Batch
import numpy as np
from scipy.stats import pearsonr
import torch
from rdkit import DataStructs
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

mut_dict, cnv_dict, gen_dict, cell_type_dict  = get_cell_feat()  # Three-dimensional array
drug_response_dict, drug_name, cell_name, drug_smiles_dict, drug_figureprints_dict = read_dr_dict()


def computing_sim_matrix(drug_figureprints_dict, gen_exp, drug_name, cell_name):
    # if os.path.exists(dict_dir + "cell_sim_matrix") and  os.path.exists(dict_dir + "drug_sim_matrix"):
    #     with open(dict_dir+ "cell_sim_matrix", 'rb') as f:
    #         cell_sim_matrix = pickle.load(f)
    #     with open(dict_dir+ "drug_sim_matrix", 'rb') as f:
    #         drug_sim_matrix = pickle.load(f)
    #     return drug_sim_matrix, cell_sim_matrix

    print('Compute drug_sim_matrix!')

    drug_sim_matrix = np.zeros((len(drug_figureprints_dict), len(drug_figureprints_dict)))
    # mi = [Chem.MolFromSmiles(drug_idx2smiles_dict[i]) for i in range(len(drug_name2idx_dict))]
    # fps = [AllChem.GetMorganFingerprint(x, 4) for x in mi]
    for i in range(len(drug_figureprints_dict)):
        for j in range(len(drug_figureprints_dict)):
            if i != j:
                drug_sim_matrix[i][j] = DataStructs.DiceSimilarity(
                    drug_figureprints_dict[drug_name[i]], drug_figureprints_dict[drug_name[j]])

    # gen_exp = gen_dict['omic_data']
    # s, _, _ = gen_exp.shape
    # gen_exp = gen_exp.reshape(s, -1)
    print('Compute cell_sim_matrix!')
    
    cell_sim_matrix = np.zeros((len(gen_exp), len(gen_exp)))
    for i in range(len(gen_exp)):
        for j in range(len(gen_exp)):
            if i != j:
                cell_sim_matrix[i][j], _ = pearsonr(gen_exp[i], gen_exp[j])
                if cell_sim_matrix[i][j] < 0:
                    cell_sim_matrix[i][j] = 0

    dict_dir = ''
    import dill
    with open(dict_dir+ "cell_sim_matrix.dill", 'wb') as f:
        dill.dump(cell_sim_matrix, f)
    with open(dict_dir+ "drug_sim_matrix.dill", 'wb') as f:
        dill.dump(drug_sim_matrix, f)

    print('Finish')

    return drug_sim_matrix, cell_sim_matrix

def computing_knn(drug_figureprints_dict, gen_exp, drug_name, cell_name, k):

    # drug_sim_matrix, cell_sim_matrix = computing_sim_matrix(drug_figureprints_dict, gen_exp, drug_name, cell_name)

    import dill
    with open('cell_sim_matrix.dill','rb') as f:
        cell_sim_matrix = dill.load(f)

    with open('drug_sim_matrix.dill','rb') as f:
        drug_sim_matrix = dill.load(f)

    cell_sim_matrix_new = np.zeros_like(cell_sim_matrix)
    for u in range(len(gen_exp)):
        v = cell_sim_matrix[u].argsort()[-k-1:-1]
        cell_sim_matrix_new[u][v] = cell_sim_matrix[u][v]

    drug_sim_matrix_new = np.zeros_like(drug_sim_matrix)
    for u in range(len(drug_figureprints_dict)):
        v = drug_sim_matrix[u].argsort()[-k-1:-1]
        drug_sim_matrix_new[u][v] = drug_sim_matrix[u][v]
    
    drug_edges = np.argwhere(drug_sim_matrix_new >  0).T
    cell_edges = np.argwhere(cell_sim_matrix_new >  0).T
    # with open(dir + "edge/drug_cell_edges_{}_knn".format(k), 'wb') as f:
    #     pickle.dump((drug_edges, cell_edges), f)

    # get cell drug response matrix


    return drug_edges, cell_edges


def fingerprint_to_numpy(fingerprint):
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def process_data(cell_gen, drug_figureprints_dict, k):

    data = dict()

    cell_name = cell_gen['cell_lines']
    drug_name = list(drug_figureprints_dict.keys())
    cell_to_idx = {cell: idx for idx, cell in enumerate(cell_name)}
    drug_to_idx = {drug: idx for idx, drug in enumerate(drug_name)}
    
    data['drug_name'] = drug_name
    data['cell_name'] = cell_name

    s, p, g = cell_gen['omic_data'].shape
    gen_exp = cell_gen['omic_data'].reshape(s, -1) # [700+, 8000+]


    data['drug_feats'] = np.array([fingerprint_to_numpy(i) for i in list(drug_figureprints_dict.values())])
    data['cell_feats'] = np.array(gen_exp)

    print('Compute knn')
    drug_edges, cell_edges = computing_knn(drug_figureprints_dict, gen_exp, drug_name, cell_name, k)

    data['drug_edges'] = drug_edges
    data['cell_edges'] = cell_edges

 

    cell_drug_edge_index = [[], []]
    cell_drug_edge_attr = []
    for entry in drug_response_dict:
        cell, drug, ic50 = entry[0], entry[1], entry[2] 

        cell_idx = cell_to_idx[cell]
        drug_idx = drug_to_idx[drug]

        cell_drug_edge_index[0].append(cell_idx)
        cell_drug_edge_index[1].append(drug_idx)
        cell_drug_edge_attr.append(ic50)

    cell_drug_edge_index = np.array(cell_drug_edge_index)

    data['cell_drug_response_edge'] = cell_drug_edge_index
    data['cell_drug_response_attr'] = cell_drug_edge_attr


    return data

def heterograph(drug_response_dict, data, data_idx):

    from torch_geometric.data import HeteroData

    '''Error'''

    graph = HeteroData()

    graph['cell_line'].x = torch.tensor(data['cell_feats'])
    graph['cell_line'].name = torch.tensor(data['cell_name'])

    graph['drug'].x = torch.from_numpy(data['drug_feats'])
    graph['drug_line'].name = torch.tensor(data['drug_name'])

    cell_to_idx = {cell: idx for idx, cell in enumerate(data['cell_name'])}
    drug_to_idx = {drug: idx for idx, drug in enumerate(data['drug_name'])}

    cell_drug_edge_index = [[], []]
    cell_drug_edge_attr = []

    
    for idx in data_idx:

        entry = drug_response_dict[idx]
        cell, drug, ic50 = entry[0], entry[1], entry[2] 

        cell_idx = cell_to_idx[cell]
        drug_idx = drug_to_idx[drug]

        cell_drug_edge_index[0].append(cell_idx)
        cell_drug_edge_index[1].append(drug_idx)
        cell_drug_edge_attr.append(ic50)

    cell_drug_edge_index = np.array(cell_drug_edge_index)
    cell_drug_edge_attr  = np.array(cell_drug_edge_attr)

    # data['cell_drug_response_edge'] = cell_drug_edge_index
    # data['cell_drug_response_attr'] = cell_drug_edge_attr

    cell_edge_index = [[], []]
    drug_edge_index = [[], []]

    for i in range(len(data['cell_edges'][0])):

        # import ipdb
        # ipdb.set_trace()

        source = data['cell_edges'][0][i]  
        target = data['cell_edges'][1][i] 
        
        if source in cell_drug_edge_index[0] and target in cell_drug_edge_index[0]:
            cell_edge_index[0].append(source)
            cell_edge_index[1].append(target)

    for i in range(len(data['drug_edges'][0])):
        source = data['drug_edges'][0][i] 
        target = data['drug_edges'][1][i] 
        
        if source in cell_drug_edge_index[1] and target in cell_drug_edge_index[1]:
            drug_edge_index[0].append(source)
            drug_edge_index[1].append(target)

    # data['cell_edge_index'] = cell_edge_index
    # data['drug_edge_index'] = drug_edge_index

    graph['cell_line', 'similar', 'cell_line'].edge_index = torch.tensor(cell_edge_index, dtype=torch.int64)
    graph['drug', 'similar', 'drug'].edge_index = torch.tensor(drug_edge_index, dtype=torch.int64)
    graph['cell_line', 'response', 'drug'].edge_index = torch.tensor(cell_drug_edge_index, dtype=torch.int64)
    graph['cell_line', 'response', 'drug'].edge_attr = torch.tensor(cell_drug_edge_attr)

    # import ipdb
    # ipdb.set_trace()

    return graph


def collate_fn(batch):

    subgraph = batch
    drug_names = subgraph['drug'].name
    cell_names = subgraph['cell'].name
        
    gen_features = subgraph['cell'].x
    # 从字典中提取药物名称对应的特征
    drug_atom = torch.cat([drug_atom_dict[name] for name in drug_names], dim=0)
    drug_bond = torch.cat([drug_bond_dict[name] for name in drug_names], dim=0)

    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)

    cnv_features = torch.cat([cnv_dict[name] for name in cell_names], dim=0)
    mut_features = torch.cat([mut_dict[name] for name in cell_names], dim=0)
    
    return {
        'subgraph': subgraph,

        'drug_atom': batched_drug_atom,
        'drug_bond': batched_drug_bond,
        
        'gen_feats': gen_features,
        'cnv_feats': cnv_features,
        'mut_feats': mut_features
    }

def graph_drp_loader(data):


    dataloader = LinkNeighborLoader(

        data,  
        edge_label_index=(('cell_line', 'response', 'drug'), data['cell_line', 'response', 'drug'].edge_index),
        edge_label=data['cell_line', 'response', 'drug'].edge_attr,
        num_neighbors=[5, 5], 
        batch_size=128,
        shuffle=True,
    )

    return dataloader


