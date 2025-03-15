import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch.utils.data import Dataset, Subset, DataLoader
from functools import partial
import os
import csv
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


from configure import args
from split import split

def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeNetGraphDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # return data
    return x, edge_index, edge_attr


class MoleculeDataset(InMemoryDataset):
    def __init__(self, args):
        self.root = args.root
        self.datasets = args.datasets
        
        # self.tokenizer = tokenizer
        # self.smiles_max_length = args.smiles_max_length
        
        self.use_mm = args.use_multimodal
        self.encoding = args.encoding
        
        self.datasets_information = self.load_datasets()
        print('datasets_information: ', len(self.datasets_information))
        all_mol_list = []
        all_label_len = 0
        for dataset_info in self.datasets_information:
            all_label_len += dataset_info['labels'].shape[1]
        print('all_label_len: ', all_label_len)
        
        label_pointer = 0
        total_idx = 0
        for dataset_info in self.datasets_information:
            print('dataset_info: ', dataset_info['task'], len(dataset_info['smiles']), dataset_info['split_type'], label_pointer, label_pointer + dataset_info['labels'].shape[1])
            for smile_idx in range(len(dataset_info['smiles'])):
                one_mol = {
                    'idx': total_idx, 
                    'smile': dataset_info['smiles'][smile_idx], 
                    'mol': dataset_info['mols'][smile_idx], 
                    'task_type': dataset_info['task_type'], 
                    'task': dataset_info['task'], 
                    'split_type': dataset_info['split_type']
                }
                one_mol['smiles_prompt'] = f"SMILES: {dataset_info['smiles'][smile_idx]}\n"
                
                if self.use_mm:
                    one_mol['smiles_prompt'] = one_mol['smiles_prompt'] + f"\nDescription of the Molecule:\n{dataset_info['description'][smile_idx]}\n"
                one_mol['cot'] = "Let's think step by step."
                one_mol['predict_prompt'] = "Please predict the " + args.datasets_property_prompt[args.datasets[0]] + " of the molecule."
                one_mol['cot_prompt'] = "Thinking process: "
                    
                one_mol_label = np.zeros(all_label_len)   # fill in the blanks of the labels with 0 in other datasets
                one_mol_label[label_pointer: label_pointer + dataset_info['labels'].shape[1]] = dataset_info['labels'][smile_idx]   # fill in the labels of the one mol where the dataset it belong to
                one_mol['label'] = one_mol_label
                all_mol_list.append(one_mol)
                total_idx = total_idx + 1
            # print('dataset info label shape: ', label_pointer, label_pointer + dataset_info['labels'].shape[1])
            label_pointer = label_pointer + dataset_info['labels'].shape[1]
        self.all_mol_list = all_mol_list  # a list of mols information in all datasets, each mol is denoted by a dictionary


    def __getitem__(self, index):
        if self.all_mol_list[index]['mol'] == None:
            return None
        
        atom_features_list, edge_index, edge_attr = mol_to_graph_data_obj_simple(self.all_mol_list[index]['mol'])

        return {
            'idx': index,
            'smile': self.all_mol_list[index]['smile'],
            'task_type': self.all_mol_list[index]['task_type'],
            'task': self.all_mol_list[index]['task'],
            'split_type': self.all_mol_list[index]['split_type'],
            'label': torch.tensor(self.all_mol_list[index]['label']),
            'smiles_prompt': self.all_mol_list[index]['smiles_prompt'],
            'cot_prompt': self.all_mol_list[index]['cot'],
            'predict_prompt': self.all_mol_list[index]['predict_prompt'],
            'atom_features_list': atom_features_list,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
        
    
    def load_datasets(self):
        datasets_information = []

        for dataset in self.datasets:
            csv_file = os.path.join(self.root, dataset, dataset + '_MM.csv')
            prop_list = []
            smiles_list = []
            mol_list = []
            description_list = []
            mol_prop = pd.read_csv(csv_file, encoding=self.encoding, index_col=0)
            task_type = ''
            dataset_info = {}

            if dataset in ['BACE', 'BBBP', 'HIV', 'ClinTox', 'Sider', 'Tox21', 'ToxCast']:   # 根据分类或回归分类
            # read CSV file
                task_type = 'classification'
                split_type = 'random'
                with open(csv_file, 'r', encoding=self.encoding) as file:
                    reader = csv.reader(file)
                    header = next(reader)  # ignore header
                    # print('smiles_index: ', smiles_index)
                    
                    if dataset == 'BACE':
                        split_type = 'scaffold'
                        smiles_index = header.index('mol')
                        classifi_index = header.index('Class')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [int(row[classifi_index])]
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                        
                    elif dataset == 'BBBP':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        classifi_index = header.index('p_np')
                        description_index = header.index('Description')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [int(row[classifi_index])]
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                            
                    elif dataset == 'ClinTox':
                        split_type = 'random'
                        smiles_index = header.index('smiles')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = []
                            for x in row[1:3]:
                                if x != '':
                                    data.append(int(x))
                                else:
                                    data.append(-10)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                            
                    elif dataset == 'HIV':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        classifi_index = header.index('HIV_active')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [int(row[classifi_index])]
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                            
                    elif dataset == 'Sider':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = []
                            for x in row[1:28]:
                                if x != '':
                                    data.append(int(x))
                                else:
                                    data.append(-10)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                    
                    elif dataset == 'Tox21':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = []
                            for x in row[0: 12]:
                                if x != '':
                                    data.append(int(float(x)))
                                else:
                                    data.append(-10)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                            
                            
                    elif dataset == 'ToxCast':
                        split_type = 'random'
                        smiles_index = header.index('smiles')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = []
                            for x in row[1:]:
                                if x != '':
                                    data.append(int(float(x)))
                                else:
                                    data.append(-10)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                        
                labels = np.array(prop_list, dtype=np.int32)
                smiles = np.array(smiles_list)
                print('smiles len: ', smiles.shape)
                print('mol len: ', len(mol_list))
                labels[labels == 0] = -1   # only 1 -1 valid，use y_valid = y_true ** 2 > 0 to judge valid
                labels[labels == -10] = 0  # substitute -10 with 0, invalid data
                dataset_info = {'task_type': task_type, 'task': dataset, 'split_type': split_type, 'smiles': smiles, 'mols': mol_list, 'labels': labels, 'description': description_list}

            elif dataset in ['QM9', 'QM8', 'QM7', 'ESOL', 'Lipo', 'FreeSolv']:
                task_type = 'regression'
                split_type = 'random'
                with open(csv_file, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)  
                    smiles_index = header.index('smiles')
                    # print('smiles_index: ', smiles_index)
                    
                    if dataset == 'QM9':
                        split_type = 'random'
                        smiles_index = header.index('smiles')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = []
                            for x in row[5:17]:
                                if x != '':
                                    data.append(float(x))
                                else:
                                    data.append(0)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                    
                    elif dataset == 'QM8':
                        split_type = 'random'
                        smiles_index = header.index('smiles')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = []
                            for x in row[1:]:
                                if x != '':
                                    data.append(float(x))
                                else:
                                    data.append(0)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                    
                    elif dataset == 'QM7':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        regress_index = header.index('u0_atom')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [float(row[regress_index])]
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                        
                    elif dataset == 'ESOL':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        regress_index = header.index('measured log solubility in mols per litre')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [float(row[regress_index])]
                            # print('ESOL: ', data)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                        
                    elif dataset == 'Lipo':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        regress_index = header.index('exp')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [float(row[regress_index])]
                            # print('Lipo: ', data)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])
                    
                    elif dataset == 'FreeSolv':
                        split_type = 'scaffold'
                        smiles_index = header.index('smiles')
                        regress_index = header.index('expt')
                        if self.use_mm:
                            description_index = header.index('Description')
                        for row in reader:
                            if len(row) == 0:
                                continue
                            smiles = row[smiles_index]
                            data = [float(row[regress_index])]
                            # print('FreeSolv: ', data)
                            mol = Chem.MolFromSmiles(smiles)
                            smiles_list.append(smiles)
                            prop_list.append(data)
                            mol_list.append(mol)
                            if self.use_mm:
                                description_list.append(row[description_index])

                    
                labels = np.array(prop_list)
                smiles = np.array(smiles_list)
                print('labels shape: ', labels.shape)
                print('smiles len: ', smiles.shape)
                print('mol len: ', len(mol_list))
                dataset_info = {'task_type': task_type, 'task': dataset, 'split_type': split_type, 'smiles': smiles, 'mols': mol_list, 'labels': labels, 'description': description_list}

            np.savez(os.path.join(self.root, dataset, dataset + '_label.npz'), data=labels)
            np.savez(os.path.join(self.root, dataset, dataset + '_smiles.npz'), smiles=smiles)
            datasets_information.append(dataset_info)
        
        return datasets_information
    
    
class MoleculeDatasetWrapper(object):
    def __init__(self, dataset, batch_size, num_gpu, valid_rate, test_rate, split_type, num_workers, split_seed):
        super(MoleculeDatasetWrapper, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.valid_rate = valid_rate
        self.test_rate = test_rate
        self.split_type = split_type
        self.seed = split_seed
        self.num_workers = num_workers
        
    def get_data_loaders(self):
        train_idx, valid_idx, test_idx = split(self.dataset.all_mol_list, self.valid_rate, self.test_rate, self.seed)
        total_idx = range(len(self.dataset.all_mol_list))
        print('train valid test idx: ', len(train_idx), len(valid_idx), len(test_idx))
        
        train_set = Subset(self.dataset, train_idx)
        valid_set = Subset(self.dataset, valid_idx)
        test_set = Subset(self.dataset, test_idx)
        total_set = Subset(self.dataset, total_idx)
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True, collate_fn=self.custom_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True, collate_fn=self.custom_collate_fn)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True, collate_fn=self.custom_collate_fn)
        total_loader = DataLoader(total_set, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True, collate_fn=self.custom_collate_fn)
        return train_loader, valid_loader, test_loader, total_loader
    
    
    def custom_collate_fn(self, batch):
        batch_data = {
            'idx': [],
            'smile': [],
            'task_type': [],
            'task': [],
            'split_type': [],
            'label': [],
            'smiles_prompt': [],
            'cot_prompt': [],
            'predict_prompt': [],
            'atom_features_list': [],
            'edge_index': [],
            'edge_attr': []
        }
        
        for item in batch:
            for key in batch_data:
                batch_data[key].append(item[key])
                
        atom_features_list = []
        edge_index_list = []
        edge_attr_list = []
        batch_list = []

        cum_node_count = 0  # use to adjust edge_index
        for i, item in enumerate(batch):
            num_nodes = item['atom_features_list'].size(0)
            atom_features_list.append(item['atom_features_list'])
            
            edge_index = item['edge_index'] + cum_node_count
            edge_index_list.append(edge_index)
            edge_attr_list.append(item['edge_attr'])
            
            # generate batch.batch
            batch_list.append(torch.full((num_nodes,), i, dtype=torch.long))
            cum_node_count += num_nodes

        atom_features = torch.cat(atom_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)  # edge index 2 x num_edges
        edge_attr = torch.cat(edge_attr_list, dim=0)
        batch = torch.cat(batch_list)
        
        batch_data['atom_features_list'] = atom_features
        batch_data['edge_index'] = edge_index
        batch_data['edge_attr'] = edge_attr
        batch_data['batch'] = batch
        batch_data['label'] = torch.cat(batch_data['label'], dim=0)
        
        return batch_data

