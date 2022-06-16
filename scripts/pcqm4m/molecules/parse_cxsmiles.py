import numpy as np
from scipy.sparse import csr_matrix
from copy import deepcopy
import networkx as nx
from rdkit.Chem.AllChem import GetAdjacencyMatrix, MolToCXSmiles, MolToSmiles, MolFromSmiles
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

from molecules.physics import get_self_euclidean_dist_matrix

def get_contact_dict(mol, angstrom_dist_threshold=3.5, path_len_threshold=5):
    mol = deepcopy(mol)
    num_atoms = mol.GetNumAtoms()

    # Get the euclidean distance matrix and adjacency matrix
    xyz = mol.GetConformer(0).GetPositions()
    dist_matrix = get_self_euclidean_dist_matrix(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    adj = GetAdjacencyMatrix(mol)

    # Get and all pair-wise graph distances between nodes
    G = nx.from_numpy_array(adj)
    path_len = nx.all_pairs_dijkstra_path_length(G, cutoff=100)
    path_len_matrix = [None] * num_atoms
    for elem in path_len:
        idx = elem[0]
        vec = [float("inf")] * num_atoms
        for key, val in elem[1].items():
            vec[key] = val
        path_len_matrix[idx] = vec
    path_len_matrix = np.array(path_len_matrix)

    # Find the index of contact based on the thresholds
    is_close = dist_matrix < angstrom_dist_threshold
    is_contact = is_close & (path_len_matrix >= path_len_threshold) & np.tri(N=path_len_matrix.shape[0], dtype=bool)
    idx_contact = np.stack(np.where(is_contact)).T

    # Convert the index of contact into a list for each atoms, with unique value for each contact
    list_contact = [[] for _ in range(num_atoms)]
    for count, contact in enumerate(idx_contact):
        list_contact[int(contact[0])].append(count)
        list_contact[int(contact[1])].append(count)

    # Write the 'contact' properties into the molecules. Useful when storing it
    for ii, val in enumerate(list_contact):
        mol.GetAtomWithIdx(ii).SetProp("contact", str(val)[1:-1].replace(", ", ";"))

    # Save all the contact properties into a dict, and return it
    out = {}
    out["num_atoms"] = num_atoms
    out["adj"] = csr_matrix(adj)
    out["is_contact"] = csr_matrix(is_contact)
    out["idx_contact"] = idx_contact
    out["num_contact"] = np.sum(is_contact)
    out["3d_dist_contact"] = dist_matrix[is_contact]
    out["path_len_contact"] = path_len_matrix[is_contact]
    out["cxsmiles"] = MolToCXSmiles(mol, allHsExplicit=True)
    out["smiles"] = MolToSmiles(mol, allHsExplicit=True)

    return out

def cxsmiles_to_mol_with_contact(cxsmiles):
    mol = MolFromSmiles(cxsmiles, sanitize=False)
    num_atoms = mol.GetNumAtoms()
    list_of_contacts = [[]] * num_atoms
    for ii, atom in enumerate(mol.GetAtoms()):
        try:
            this_list = [int(val) for val in atom.GetProp("contact").split(";")]
            list_of_contacts[ii] = this_list
        except: pass
    max_count = max(sum(list_of_contacts, []) + [-1]) + 1

    if max_count == 0:
        contact_idx = np.zeros(shape=(0,2), dtype=int)
    else:
        contact_idx = []
        for this_count in range(max_count):
            this_count_found = []
            for ii in range(num_atoms):
                if this_count in list_of_contacts[ii]:
                    this_count_found.append(ii)
            contact_idx.append(this_count_found)
        contact_idx = np.array(contact_idx, dtype=int)

    return mol, contact_idx


def cxsmiles_to_graph_with_contact(cxsmiles):
    mol, contact_idx = cxsmiles_to_mol_with_contact(cxsmiles)
    graph = mol2graph(mol)
    graph["contact_idx"] = contact_idx
    return graph


def mol2graph(mol):
    """
    Slightly modified from ogb `smiles2graph`. Takes mol instead of smiles.

    Converts rdkit.Mol string to graph Data object
    :input: rdkit.Mol
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph

def try_read_cxsmiles(cxs, ii):
    try:
        graph = cxsmiles_to_graph_with_contact(cxs)
    except:
        print(f"Failed at {ii} for {cxs}")
        graph = None
    return graph

