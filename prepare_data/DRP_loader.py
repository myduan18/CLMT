import sys
import random
from prepare_data.create_cell_feat import *
from prepare_data.create_drug_feat import *
from prepare_data.create_drp_dict import *
from binarization_drp import *
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader


def split_drug_set(seed = 0):
    random.seed(seed)
    train_portion = 0.2
    num_drug = drug_name.shape[0]
    test_drug_id = random.sample(range(num_drug), int(train_portion*num_drug))
    train_drug_id = list(set([i for i in range(num_drug)])-set(test_drug_id))
    return train_drug_id,test_drug_id



def computing_sim_matrix(drug_figureprints_dict, gen_exp, drug_name, cell_name, dict_dir):

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

    import dill
    with open(dict_dir+ "cell_sim_matrix.dill", 'wb') as f:
        dill.dump(cell_sim_matrix, f)
    with open(dict_dir+ "drug_sim_matrix.dill", 'wb') as f:
        dill.dump(drug_sim_matrix, f)

    print('Finish')

    return drug_sim_matrix, cell_sim_matrix

def computing_knn(drug_figureprints_dict, gen_exp, drug_name, cell_name, k, dataset):

    root = os.getcwd()

    if dataset == 'CCLE':
        dict_dir = root + '/CCLE/DRP_dataset/'
    elif dataset == 'GDSC':
        dict_dir = root + '/GDSC/DRP_dataset/'

    # dict_dir = root + '/CCLE/DRP_dataset/'

    # drug_sim_matrix, cell_sim_matrix = computing_sim_matrix(drug_figureprints_dict, gen_exp, drug_name, cell_name, dict_dir)

    import dill
    with open(dict_dir+'cell_sim_matrix.dill','rb') as f:
        cell_sim_matrix = dill.load(f)

    with open(dict_dir+'drug_sim_matrix.dill','rb') as f:
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
    print("finish loading similarity matrix!")

    return drug_edges, cell_edges


def fingerprint_to_numpy(fingerprint):
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def process_data(gen_dict, mut_dict, cnv_dict, drug_figureprints_dict, dataset, k):

    data = dict()

    cell_name = [i for i in gen_dict.keys()]
    drug_name = list(drug_figureprints_dict.keys())

    cell_to_idx = {cell: idx for idx, cell in enumerate(cell_name)}
    drug_to_idx = {drug: idx for idx, drug in enumerate(drug_name)}
    
    data['cell_name'] = cell_name
    data['drug_name'] = drug_name 

    # s, p, g = cell_gen['omic_data'].shape
    # gen_exp = cell_gen['omic_data'].reshape(s, -1) # [700+, 8000+]
    gen_exp = np.array([i for i in gen_dict.values()])

    mut_data = []
    cnv_data = []
    for c in cell_name:
        mut_data.append(mut_dict[c])
        cnv_data.append(cnv_dict[c])
    mut_data = np.array(mut_data)
    cnv_data = np.array(cnv_data)

    # omic_data = np.concatenate((gen_exp, mut_data, cnv_data), axis=1) # concat multi-omic data

    data['drug_feats'] = np.array([fingerprint_to_numpy(i) for i in list(drug_figureprints_dict.values())])
    # data['cell_feats'] = np.array(gen_exp)
    data['cell_feats'] = gen_exp

    print('Compute knn')
    drug_edges, cell_edges = computing_knn(drug_figureprints_dict, gen_exp, drug_name, cell_name, k, dataset)

    data['drug_edges'] = drug_edges
    data['cell_edges'] = cell_edges


    return data, cell_name, drug_name




def graph_drp_nan_dataset(drug_response_nan_dict, data, batch_size):


    def truncate_list(input_list):
        length = len(input_list)
        remainder = length % batch_size

        if remainder < 10:
            return input_list[:length - remainder]
        return input_list
    
    data_idx = list(range(len(drug_response_nan_dict)))

    from torch_geometric.data import HeteroData

    '''Error'''

    graph = HeteroData()

    # ipdb.set_trace()

    graph['cell_line'].x = torch.tensor(data['cell_feats']).float()
    graph['cell_line'].name = torch.tensor(data['cell_name'])

    graph['drug'].x = torch.from_numpy(data['drug_feats']).float()
    graph['drug'].name = torch.tensor(data['drug_name'])

    cell_to_idx = {cell: idx for idx, cell in enumerate(data['cell_name'])}
    drug_to_idx = {drug: idx for idx, drug in enumerate(data['drug_name'])}

    graph['cell_line'].n_id = torch.tensor([cell_to_idx[i] for i in data['cell_name']])
    graph['drug'].n_id = torch.tensor([drug_to_idx[i] for i in data['drug_name']])

    cell_drug_edge_index = [[], []]
    cell_drug_edge_attr = []

    for idx in data_idx:

        entry = drug_response_nan_dict[idx]
        cell, drug, ic50 = entry[0], entry[1], entry[2] 

        # if cell == 908455 and drug == 1401:

        cell_idx = cell_to_idx[cell]
        drug_idx = drug_to_idx[drug]
        
        cell_drug_edge_index[0].append(cell_idx)
        cell_drug_edge_index[1].append(drug_idx)
        cell_drug_edge_attr.append(ic50)


    cell_drug_edge_index = np.array(cell_drug_edge_index)
    cell_drug_edge_attr  = np.array(cell_drug_edge_attr)

    # import ipdb
    # ipdb.set_trace()

    cell_edge_index = [[], []]
    drug_edge_index = [[], []]

    for i in range(len(data['cell_edges'][0])):

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

    # graph['cell_line', 'similar', 'cell_line'].edge_index = torch.tensor(cell_edge_index, dtype=torch.int64)
    # graph['drug', 'similar', 'drug'].edge_index = torch.tensor(drug_edge_index, dtype=torch.int64)

    graph['cell_line', 'similar', 'cell_line'].edge_index = torch.tensor(data['cell_edges'], dtype=torch.int64)
    graph['drug', 'similar', 'drug'].edge_index = torch.tensor(data['drug_edges'], dtype=torch.int64)

    graph['cell_line', 'response', 'drug'].edge_index = torch.tensor(cell_drug_edge_index, dtype=torch.int64)
    graph['cell_line', 'response', 'drug'].edge_attr = torch.tensor(cell_drug_edge_attr) 

    return graph


def graph_drp_dataset(drug_response_dict, data, data_idx, batch_size):

    # ipdb.set_trace()

    def truncate_list(input_list):
        length = len(input_list)
        remainder = length % batch_size

        # 如果余数小于5，舍弃最后的几个元素
        if remainder < 10:
            return input_list[:length - remainder]
        return input_list
    
    data_idx = truncate_list(data_idx)
    
    from torch_geometric.data import HeteroData


    graph = HeteroData()

    # ipdb.set_trace()

    graph['cell_line'].x = torch.tensor(data['cell_feats']).float()
    graph['cell_line'].name = torch.tensor(data['cell_name'])

    graph['drug'].x = torch.from_numpy(data['drug_feats']).float()
    graph['drug'].name = torch.tensor(data['drug_name'])

    cell_to_idx = {cell: idx for idx, cell in enumerate(data['cell_name'])}
    drug_to_idx = {drug: idx for idx, drug in enumerate(data['drug_name'])}

    graph['cell_line'].n_id = torch.tensor([cell_to_idx[i] for i in data['cell_name']])
    graph['drug'].n_id = torch.tensor([drug_to_idx[i] for i in data['drug_name']])

    cell_drug_edge_index = [[], []]
    cell_drug_edge_attr = []

    for idx in data_idx:

        entry = drug_response_dict[idx]
        cell, drug, ic50 = entry[0], entry[1], entry[2] 

        # if cell == 908455 and drug == 1401:

        cell_idx = cell_to_idx[cell]
        drug_idx = drug_to_idx[drug]
        
        cell_drug_edge_index[0].append(cell_idx)
        cell_drug_edge_index[1].append(drug_idx)
        cell_drug_edge_attr.append(ic50)


    cell_drug_edge_index = np.array(cell_drug_edge_index)
    cell_drug_edge_attr  = np.array(cell_drug_edge_attr)

    # import ipdb
    # ipdb.set_trace()

    cell_edge_index = [[], []]
    drug_edge_index = [[], []]

    for i in range(len(data['cell_edges'][0])):

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

    # graph['cell_line', 'similar', 'cell_line'].edge_index = torch.tensor(cell_edge_index, dtype=torch.int64)
    # graph['drug', 'similar', 'drug'].edge_index = torch.tensor(drug_edge_index, dtype=torch.int64)

    graph['cell_line', 'similar', 'cell_line'].edge_index = torch.tensor(data['cell_edges'], dtype=torch.int64)
    graph['drug', 'similar', 'drug'].edge_index = torch.tensor(data['drug_edges'], dtype=torch.int64)

    graph['cell_line', 'response', 'drug'].edge_index = torch.tensor(cell_drug_edge_index, dtype=torch.int64)
    graph['cell_line', 'response', 'drug'].edge_attr = torch.tensor(cell_drug_edge_attr) 

    return graph



def graph_drp_loader(data, batch_size, shuffle=True):

    from torch_geometric.loader import LinkNeighborLoader

    edge_label_index = (('cell_line', 'response', 'drug'), data['cell_line', 'response', 'drug'].edge_index)

    dataloader = LinkNeighborLoader(

        data,  
        edge_label_index= edge_label_index,
        edge_label = data['cell_line', 'response', 'drug'].edge_attr,
        # num_neighbors=[5, 5], 
        num_neighbors={
                        ('cell_line', 'response', 'drug'): [8, 4],
                        ('cell_line', 'similar', 'cell_line'): [6, 3],
                        ('drug', 'similar', 'drug'): [6, 3]
            },  
        # num_neighbors={('cell_line', 'response', 'drug'): [10]},
        batch_size=batch_size,
        shuffle=shuffle,
    )


    return dataloader
