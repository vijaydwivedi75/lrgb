from rdkit.Chem.AllChem import GetAdjacencyMatrix, MolToSmiles, SDMolSupplier, RemoveHs, MolFromSmiles
from tqdm import tqdm

def sdf_to_mols(sdf_file, max_mols=None, sanitize=True, removeHs=True, strictParsing=True):
    mols = []
    with SDMolSupplier(sdf_file, sanitize=sanitize, removeHs=removeHs, strictParsing=strictParsing) as suppl:
        for mol in tqdm(suppl):
            mols.append(mol)
            if (max_mols is not None) and (len(mols) >= max_mols):
                break
    return mols

def get_mol_props(mol):
    out = {}
    out["atom_list"] = [atom.GetSymbol() for atom in mol.GetAtoms()]
    out["adj_with_bonds"] = GetAdjacencyMatrix(mol, useBO=True)
    out["adj"] = np.clip(out["adj_with_bonds"], a_max=1)
    out["smiles"] = MolToSmiles(mol)

    return out

def get_num_atoms_from_smiles(s):
    num = RemoveHs(MolFromSmiles(s)).GetNumAtoms(onlyExplicit=True)
    return num
