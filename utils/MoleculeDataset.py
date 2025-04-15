import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tdc.single_pred import Tox


import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from utils import one_hot_encoding, get_atom_features, get_bond_features, create_scaffold_split, balanced, label_dist

from torch_geometric.data import Data


class MoleculeDataset(Dataset):
    """hERG Molecule dataset."""

    def __init__(self, data_split='Train'):
        """
        Arguments:
            data_split (string): 'Train', 'Valid', 'Test', 'Cal'
            root_dir (string): Directory with all the images.
        """
        self.data = Tox(name = 'herg_karim')

        df = self.data.get_data(format = 'df')

        # get the splits
        self.split = create_scaffold_split(df, seed = 254, frac = [0.7, 0.1, 0.1, 0.1], entity = self.data.entity1_name)

        # balance the data
        # self.split_balanced = balanced(self.split[data_split], oversample=True, seed=254)

        self.data_split = data_split

        self.drug = self.split[data_split]['Drug']
        self.label = self.split[data_split]['Y']

        self.num_classes = 2

    def __len__(self):
        return len(self.label) - 1

    def __getitem__(self, idx):
        # approach inspired by https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/

        sample = self.drug[idx]

        label = self.label[idx]

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(sample)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()

        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype = torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i,j)) in enumerate(zip(rows, cols)):

            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

        EF = torch.tensor(EF, dtype = torch.float)

        # construct label tensor
        y_tensor = torch.tensor(np.array([label]), dtype = torch.long)

        # construct Pytorch Geometric data object
        data = Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor)

        return data


    def plot_labels(self, output_dir):
        """
        Function to plot the label distributions for each data split

        :param output_dir: output directory for the files '{data_split}_dist.png' to be saved
        :type output_dir: string
        :return: None
        :rtype:
        """

        file = os.path.join(output_dir, self.data_split + '_dist.png')
        label_dist(self.split[self.data_split].Y, file, name = " Label Distribution for " + self.data_split + " Set")

    def plot_similarity(self):
        """
        Assess the structural similarity of the molecules within each data split.

        TODO: change this so that the similarity scores for 2 of the same molecules are not calculated

        :return: None
        :rtype:
        """

        def apply_morgan(x):
            mol = Chem.MolFromSmiles(x)
            return Chem.RDKFingerprint(mol)

        def get_similarity(x, y):
            print(type(x))
            return DataStructs.FingerprintSimilarity(x, y)

        for s in ['train', 'valid', 'test', 'cal']:

            # get the molecule fingerprint
            fingerprints = self.split[s]['Drug'].apply(apply_morgan)

            # calculate the similarity
            dists = np.vectorize(get_similarity)(fingerprints[:-1], fingerprints[1:])

            # create a histogram of the result
            plt.hist(dists, bins=10, color='lightgreen')

            # set the title and axis labels
            plt.title('Histogram of Similarity Scores for ' + s)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.show()



