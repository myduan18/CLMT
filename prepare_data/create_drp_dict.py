import math
import os
import random
import numpy as np
import pandas as pd
import os
import sys
import ipdb
sys.path.append("prepare_data")

def create_drp_dict(path_ic50, path_drug_id_name, path_ccle_id_name, save_path):


    def extract_and_convert(input_string):
        digits = ''.join(char for char in input_string if char.isdigit())
        
        first_non_zero_index = -1
        for i, digit in enumerate(digits):
            if digit != '0':
                first_non_zero_index = i
                break
        
        if first_non_zero_index != -1:
            extracted_digits = digits[first_non_zero_index:]
            return int(extracted_digits)
        else:
            return 0

    IC50table = pd.read_csv(path_ic50)
    # IC50table = IC50table.rename(columns={'0': 'cell_id'})
    # IC50table = IC50table.set_index('cell_id')
    # drug_id_name = pd.read_csv(path_drug_id_name, header=None)
    # drug_id_name.rename(columns={0: 'id', 1: 'name'}, inplace=True)

    drug_id_name = pd.read_csv(path_drug_id_name, header=0)
    ccle_id_name = pd.read_csv(path_ccle_id_name,  sep='\t')
    

    def re_drug(x): return drug_id_name[drug_id_name['name'] == x]['CID'].item()
    def re_ccle(x): return extract_and_convert(ccle_id_name[ccle_id_name['CCLE_ID'] == x]['depMapID'].item())
    
    IC50table.rename(columns={x: re_drug(x)
                              for x in IC50table.columns}, inplace=True)
    IC50table.rename(index={x: re_ccle(x)
                              for x in IC50table.index}, inplace=True)
    
    # IC50table.rename(columns={x: drug_id_name[drug_id_name['name'] == x]['CID'].item()
    # import ipdb
    # ipdb.set_trace()

    cell_drug_interaction = [[cell, drug]
                             for cell in IC50table.index for drug in IC50table.columns]
    
    # Remove all the nan value
    # drug_response_dict = [[int(cell), int(drug), IC50table[drug][cell], 1/(1+math.exp(IC50table[drug][cell])**(-0.1))]
    #                       for [cell, drug] in cell_drug_interaction if not np.isnan(IC50table[drug][cell])]

    drug_response_dict = [[cell, drug, IC50table[drug][cell], 1/(1+math.exp(IC50table[drug][cell])**(-0.1))]
                            for [cell, drug] in cell_drug_interaction if not np.isnan(IC50table[drug][cell])]
    
    drug_name = IC50table.columns
    cell_name = IC50table.index
    def save_dr_dict(save_path):
        
        np.save(os.path.join(save_path, 'drug_response.npy'), drug_response_dict)
        np.save(os.path.join(save_path, 'drug_name.npy'), drug_name)
        np.save(os.path.join(save_path, 'cell_name.npy'), cell_name)
        print("finish saving drug response data!")
    save_dr_dict(save_path)

    return drug_response_dict, drug_name, cell_name


# def read_dr_dict():
#     drug_response_dict = np.load(root + '/Data/DRP_dataset/drug_response.npy',allow_pickle=True)
#     drug_name = np.load(root + '/Data/DRP_dataset/drug_name.npy',allow_pickle=True)
#     cell_name = np.load(root + '/Data/DRP_dataset/cell_name.npy',allow_pickle=True)
#     return drug_response_dict, drug_name, cell_name


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):


    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.warning')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    # arr = np.zeros((1,), dtype=np.int8)
    # Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)

    return fingerprint


def read_dr_dict():

    root = os.getcwd()

    #############################    CCLE     #############################
    # drug_response_dict = np.load(root + '/CCLE/DRP_dataset/drug_response.npy',allow_pickle=True)
    # drug_name = np.load(root + '/CCLE/DRP_dataset/drug_name.npy',allow_pickle=True)
    # cell_name = np.load(root + '/CCLE/DRP_dataset/cell_name.npy',allow_pickle=True)

    # mixed_array = np.empty(drug_response_dict.shape, dtype=object)
    # mixed_array[:, 0] = drug_response_dict[:, 0]
    # mixed_array[:, 1:4] = drug_response_dict[:, 1:4].astype(float)

    # drug_response_dict = mixed_array
    # drug_smiles = pd.read_csv(root + '/CCLE/CCLE_data/CCLE_drug_id_name.csv', header=0)


    #############################    GDSC     #############################
    drug_response_dict = np.load(root + '/GDSC/DRP_dataset/drug_response.npy',allow_pickle=True)
    drug_name = np.load(root + '/GDSC/DRP_dataset/drug_name.npy',allow_pickle=True)
    cell_name = np.load(root + '/GDSC/DRP_dataset/cell_name.npy',allow_pickle=True)

    '''Delete the drug without smiles'''
    drug_smiles = pd.read_csv(root + '/GDSC/drug_descriptions.csv', encoding='latin-1', index_col=0)
    
    #############################    GDSC     #############################

    drug_response_dict = drug_response_dict[np.isin(drug_response_dict[:, 1], list(drug_smiles.CID.values))]
    # drug_name = np.unique(drug_response_dict[:,1])
    # cell_name = np.unique(drug_response_dict[:,0])
    
    drug_with_ID = [i for i in drug_name if i in list(drug_smiles.CID.values)]


    assert len(drug_with_ID) == len(drug_name)

    drug_smiles_dict = drug_smiles.loc[drug_smiles.CID.isin(drug_with_ID), :].set_index('CID').T.to_dict()
 
    drug_figureprints_dict = {i: smiles_to_fingerprint(drug_smiles_dict[i]['smiles']) for i in drug_with_ID}

    # ipdb.set_trace()
    print("finish loading drug response data!")
    
    return drug_response_dict, drug_name, cell_name, drug_smiles_dict, drug_figureprints_dict


def create_drp_set(type='mix', drug_name = None, cell_name = None, drug_response_dict = None, seed = 1024): ## type: mix, cb, db
    random.seed(seed)
    train_portion = 0.2
    num_cell = cell_name.shape[0]
    num_drug = drug_name.shape[0]
    num_total = len(drug_response_dict)
    test_cell_id = random.sample(range(num_cell), int(train_portion*num_cell))
    test_cell = cell_name[test_cell_id]
    test_drug_id = random.sample(range(num_drug), int(train_portion*num_drug))
    test_drug = drug_name[test_drug_id]
    test_id = random.sample(range(num_total), int(train_portion*num_total))
    train_id = list(set([i for i in range(num_total)])-set(test_id))
    # Create mixed set:
    if type == 'mix':
        train_idx = train_id
        test_idx = test_id
    # Create cell blind set:
    elif type == 'cb':
        train_idx = [idx for idx, [cell, drug,
                                             ic50, norm_ic50] in enumerate(drug_response_dict) if cell not in test_cell]
        test_idx = [idx for idx,[cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if cell in test_cell]  
    # Create drug blind set:
    elif type == 'db':
        train_idx = [idx for idx, [cell, drug,
                                             ic50,norm_ic50] in enumerate(drug_response_dict) if drug not in test_drug]
        test_idx = [idx for idx, [cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if drug in test_drug]
    return train_idx, test_idx


def n_fold_split(type='mix', drug_name = None, cell_name = None, drug_response_dict = None, seed = 1234, n_folds = 10, test_portion = 0.2): ## type: mix, cb, db
   random.seed(seed)
   print("Train type is : ",{type})
   if type == 'mix':
        
        seed = 2048
        random.seed(seed)
        np.random.seed(seed)
        num_total = len(drug_response_dict) 
        indices = np.arange(num_total)
        np.random.shuffle(indices)
        test_size = int(num_total * test_portion)
        train_val_size = num_total - test_size 
        test_idx = indices[:test_size] 
        train_val_indices = indices[test_size:] 
        fold_size = train_val_size // n_folds 
        train_dict = { } 
        val_dict = {}
        for i in range(5):
            validation_indices = train_val_indices[i * fold_size:(i+1) * fold_size]
            train_indices = np.concatenate((train_val_indices[:i * fold_size], train_val_indices[(i + 1) * fold_size:]))
            train_dict[i] = list(train_indices)
            val_dict[i] = list(validation_indices)

        return train_dict, val_dict, test_idx
   
   elif type == 'cb':
        random.seed(seed)
        np.random.seed(seed)
        num_cells = len(cell_name)
        cell_indices = np.arange(num_cells)
        np.random.shuffle(cell_indices)
        test_size = int(num_cells * test_portion)
        train_val_size = num_cells - test_size
        test_cell_indices = cell_indices[:test_size]
        cell_folds = np.array_split(cell_indices[test_size:], n_folds)
        test_idx = [idx for idx,[cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[test_cell_indices]]
        train_dict = { } 
        val_dict = {}
        for i in range(5):
            val_cell_indices = cell_folds[i]
            train_cell_indices = np.concatenate([cell_folds[j] for j in range(n_folds) if j != i])
            train_idx = [idx for idx, [cell, drug,
                                                    ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[train_cell_indices]]
            val_idx = [idx for idx, [cell, drug,
                                                    ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[val_cell_indices]]
            train_dict[i] = train_idx
            val_dict[i] = val_idx

            # ipdb.set_trace()

        return train_dict, val_dict, test_idx
   
   elif type == 'db':
        
        # seed = 128
        # according to the drug name, extracting the drug smiles and the drug scaffold_split

        df = pd.read_csv(root + '/Data/drug_descriptions_synonyms.csv', encoding='latin-1', index_col=0)
        # df = pd.read_csv('/home/hai/hai_disk/Mia/Hi-GeoMVP-dmy/CCLE/CCLE_data/CCLE_drug_id_name.csv', encoding='latin-1', index_col=0)

        # import ipdb
        # ipdb.set_trace()

        id2smiles = dict(zip(df['CID'], df['smiles']))

        smiles_list = [id2smiles[i] for i in id2smiles.keys() if i in drug_name]

        from scaffold_split import scaffold_split  

        train_dict = { } 
        val_dict = {}
        test_dict = {}

        for i in range(5):

            train_drug_indices, val_drug_indices, test_drug_indices = scaffold_split(
                smiles_list, 
                drug_name,
                train_size=0.72, 
                valid_size=0.08, 
                test_size=0.2,
                seed = seed,
                include_chirality = False,
                balanced = False
            )

            train_idx = [idx for idx, [cell, drug,ic50,norm_ic50] in enumerate(drug_response_dict) 
                        if drug in drug_name[train_drug_indices]]
            val_idx = [idx for idx, [cell, drug,ic50,norm_ic50] in enumerate(drug_response_dict) 
                        if drug in drug_name[val_drug_indices]]
            test_idx = [idx for idx,[cell,drug, ic50,norm_ic50] in enumerate(drug_response_dict) 
                        if drug in drug_name[test_drug_indices]] 
            
            train_dict[i] = train_idx
            val_dict[i] = val_idx
            test_dict[i] = test_idx

            # import ipdb

            # ipdb.set_trace()
        return train_dict, val_dict, test_dict

if __name__ =='__main__':

    root = os.getcwd()

    save_path = root + '/CCLE/DRP_dataset'
    path_ic50 = root + '/CCLE/CCLE_data/Table_S19_CCLE_Drug_response_IC50.csv'
    path_drug_id_name = root + '/CCLE/CCLE_data/CCLE_drug_id_name.csv'
    path_ccle_id_name = root + '/CCLE/CCLE_data/Cell_lines_annotations_20181226.txt'

    create_drp_dict(path_ic50, path_drug_id_name, path_ccle_id_name, save_path)

    read_dr_dict()