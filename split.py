from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from collections import defaultdict
import random
import numpy as np


def generate_scaffold(mol, include_chirality=False):
    # print('scaffold, mol: ', mol)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_ids(dataset):
    scaffolds = defaultdict(set)
    for i, mol in enumerate(dataset):
        if mol['mol'] == None:
            continue
        scaffold = generate_scaffold(mol['mol'])   # the mol in the dataset list is a dict, save the information not only the mol objects
        scaffolds[scaffold].add(mol['idx'])   # can not add i, because there are random idxs, must save the smile idx of the mol
    return scaffolds


def scaffold_split(dataset, valid_rate, test_rate, seed, balanced=False):
    train_ids, valid_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    total_num = len(dataset)
    train_size = (1 - valid_rate - test_rate) * total_num
    valid_size = valid_rate * total_num
    test_size = test_rate * total_num

    scaffold_to_indices = scaffold_to_ids(dataset)
    if balanced:
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:
        # sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(valid_ids) + len(index_set) <= valid_size:
            valid_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    return train_ids, valid_ids, test_ids


def split(dataset, valid_rate, test_rate, seed):
    random_split_indices_list = []
    scaffold_split_mol_list = []
    
    for mol in dataset:
        # 3 print('mol task split_type: ', mol['task'], mol['split_type'])
        if mol['mol'] == None:
            continue
        if mol['split_type'] == 'random':
            random_split_indices_list.append(mol['idx'])
        elif mol['split_type'] == 'scaffold':
            scaffold_split_mol_list.append(mol)
        else:
            raise ValueError(f'split_type not supported.')
    
    # random split the datasets with a split_type of 'random'
    random_num = len(random_split_indices_list)
    np.random.seed(seed)
    np.random.shuffle(random_split_indices_list)
    random_valid_size = int(np.floor(valid_rate * random_num))
    random_test_size = int(np.floor(test_rate * random_num))
    random_test_idx, random_valid_idx, random_train_idx = random_split_indices_list[: random_test_size], random_split_indices_list[random_test_size: random_test_size + random_valid_size], random_split_indices_list[random_test_size + random_valid_size:]


    scaffold_train_idx, scaffold_valid_idx, scaffold_test_idx = scaffold_split(scaffold_split_mol_list, valid_rate, test_rate, seed, balanced=True)
    
    train_idx = random_train_idx + scaffold_train_idx
    valid_idx = random_valid_idx + scaffold_valid_idx
    test_idx = random_test_idx + scaffold_test_idx

    return train_idx, valid_idx, test_idx

