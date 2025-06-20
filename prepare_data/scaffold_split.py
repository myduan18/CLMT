import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Union
import random


def generate_scaffold(mol: Chem.Mol, include_chirality: bool = False) -> str:
    """
    Generate the Murcko skeleton from the molecule. 
    Parameters:
    mol: RDKit molecule object
    include_chirality: Whether to include chirality information in the scaffold 
    Return:
    The SMILES string of the skeleton 
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    return scaffold_smiles


def scaffold_split(
    dataset: List[Union[str, Chem.Mol]],
    drug_names: List[str] = None,  
    smiles_list: List[str] = None,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    include_chirality: bool = False,
    balanced: bool = False
) -> Tuple[List[int], List[int], List[int], List[str], List[str], List[str]]:  
    """
        Divide the dataset based on the molecular skeleton structure and return the corresponding drug names at the same time. 
        Parameters:
        dataset: List of molecules, which can be a list of SMILES strings or a list of RDKit molecule objects.
        drug_names: List of drug names, corresponding one-to-one with the dataset.
        smiles_list: If the dataset is a list of molecule objects, a corresponding list of SMILES can be provided (for record-keeping).
        train_size: Proportion of the training set.
        valid_size: Proportion of the validation set.
        test_size: Proportion of the test set.
        seed: Random seed.
        include_chirality: Whether to include chirality information in the scaffold.
        balanced: Whether to perform balanced partitioning (ensuring that scaffolds of similar sizes are distributed across all sets). 
        Return:
        Index lists of the training set, validation set, and test set, as well as the corresponding lists of drug names. 
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if isinstance(dataset[0], str):
        mols = [Chem.MolFromSmiles(smiles) for smiles in dataset]
    else:
        mols = dataset
        
    if smiles_list is None and isinstance(dataset[0], Chem.Mol):
        smiles_list = [Chem.MolToSmiles(mol) for mol in dataset]
    
    if drug_names is None:
        drug_names = [f"Compound_{i}" for i in range(len(dataset))]
    
    valid_indices = []
    valid_mols = []
    for i, mol in enumerate(mols):
        if mol is not None:
            valid_indices.append(i)
            valid_mols.append(mol)
    
    scaffolds = {}
    for i, mol in zip(valid_indices, valid_mols):
        scaffold = generate_scaffold(mol, include_chirality)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [i]
        else:
            scaffolds[scaffold].append(i)
    
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(
        scaffolds.items(), key=lambda x: (len(x[1]), x[0]), reverse=True)]
    
    train_cutoff = train_size
    valid_cutoff = train_cutoff + valid_size
    
    if balanced:
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in scaffold_sets:
            if len(scaffold_set) == 1:
                p = np.random.random()
                if p < train_cutoff:
                    train_idx.extend(scaffold_set)
                elif p < valid_cutoff:
                    valid_idx.extend(scaffold_set)
                else:
                    test_idx.extend(scaffold_set)
            else:
                n_train = int(len(scaffold_set) * train_size)
                n_valid = int(len(scaffold_set) * valid_size)
                
                scaffold_set_perm = scaffold_set.copy()
                random.shuffle(scaffold_set_perm)
                
                train_idx.extend(scaffold_set_perm[:n_train])
                valid_idx.extend(scaffold_set_perm[n_train:n_train+n_valid])
                test_idx.extend(scaffold_set_perm[n_train+n_valid:])
    else:
        train_cutoff_index = int(train_cutoff * len(valid_indices))
        valid_cutoff_index = int(valid_cutoff * len(valid_indices))
        
        train_idx, valid_idx, test_idx = [], [], []
        
        idx_assigned = 0
        for scaffold_set in scaffold_sets:
            if idx_assigned + len(scaffold_set) <= train_cutoff_index:
                train_idx.extend(scaffold_set)
            elif idx_assigned + len(scaffold_set) <= valid_cutoff_index:
                valid_idx.extend(scaffold_set)
            else:
                test_idx.extend(scaffold_set)
            idx_assigned += len(scaffold_set)
    
    # train_drug_names = [drug_names[i] for i in train_idx]
    # valid_drug_names = [drug_names[i] for i in valid_idx]
    # test_drug_names = [drug_names[i] for i in test_idx]

    # import ipdb
    # ipdb.set_trace()
    
    return train_idx, valid_idx, test_idx