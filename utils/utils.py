import sys
import pandas as pd
import numpy as np
from rdkit import Chem


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom,
                      use_chirality=True,
                      hydrogens_implicit=True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond,
                      use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_scaffold_split(df, seed, frac, entity):
    # reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except:
        raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")
    from tqdm import tqdm
    from random import Random

    from collections import defaultdict
    random = Random(seed)

    s = df[entity].values
    scaffolds = defaultdict(set)
    idx2mol = dict(zip(list(range(len(s))), s))

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=False)
            scaffolds[scaffold].add(i)
        except:
            print(smiles + ' returns RDKit error and is thus omitted...')
            error_smiles += 1

    train, val, test, cal = [], [], [], []
    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    cal_size = int((len(df) - error_smiles) * frac[2])
    test_size = (len(df) - error_smiles) - train_size - val_size - cal_size
    train_scaffold_count, val_scaffold_count, cal_scaffold_count, test_scaffold_count = 0, 0, 0, 0

    # index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1

            elif len(cal) + len(index_set) < cal_size:
                cal += index_set
                cal_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {'train': df.iloc[train].reset_index(drop=True),
            'valid': df.iloc[val].reset_index(drop=True),
            'cal': df.iloc[cal].reset_index(drop=True),
            'test': df.iloc[test].reset_index(drop=True)}


def label_dist(y, filename, name = None):

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except:
        utils.install("seaborn")
        utils.install("matplotlib")
        import seaborn as sns
        import matplotlib.pyplot as plt

    median = np.median(y)
    mean = np.mean(y)

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.15, 1)})

    if name is None:
        sns.boxplot(y, ax=ax_box).set_title("Label Distribution")
    else:
        sns.boxplot(y, ax=ax_box).set_title("Label Distribution of " + str(name) + " Dataset")
    ax_box.axvline(median, color='b', linestyle='--')
    ax_box.axvline(mean, color='g', linestyle='--')
    ax_hist.axvline(median, color='b', linestyle='--')
    ax_hist.axvline(mean, color='g', linestyle='--')
    ax_hist.legend({'Median':median,'Mean':mean})
    sns.distplot(y, ax=ax_hist)
    ax_box.set(xlabel='')
    plt.savefig(filename)


def balanced(val, oversample=False, seed=42):

    class_ = val.Y.value_counts().keys().values
    major_class = class_[0]
    minor_class = class_[1]

    if not oversample:
        print(
            " Subsample the majority class is used, if you want to do "
            "oversample the minority class, set 'balanced(oversample = True)'. ",
            flush=True, file=sys.stderr)
        val = pd.concat(
            [val[val.Y == major_class].sample(
                n=len(val[val.Y == minor_class]), replace=False,
                random_state=seed), val[val.Y == minor_class]]).sample(
            frac=1,
            replace=False,
            random_state=seed).reset_index(
            drop=True)
    else:
        print(" Oversample of minority class is used. ", flush=True,
              file=sys.stderr)
        val = pd.concat(
            [val[val.Y == minor_class].sample(
                n=len(val[val.Y == major_class]), replace=True,
                random_state=seed), val[val.Y == major_class]]).sample(
            frac=1,
            replace=False,
            random_state=seed).reset_index(
            drop=True)
    return val
