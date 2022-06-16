import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import GetAdjacencyMatrix, MolToSmiles
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcSpherocityIndex, CalcPBF

EPS = 1e-6

this_path = os.path.dirname(os.path.realpath(__file__))
PERIODIC_TABLE = pd.read_csv(os.path.join(this_path, "periodic_table.csv"))
PERIODIC_TABLE = PERIODIC_TABLE.set_index("AtomicNumber")

def translate_to_cofm(masses, xyz):
    # Position of centre of mass in original coordinates
    cofm = sum(masses[:,np.newaxis] * xyz) / np.sum(masses)
    # Transform to CofM coordinates and return
    xyz -= cofm
    return xyz

def get_inertia_matrix(masses, xyz):
    # Moment of intertia tensor
    xyz = translate_to_cofm(masses, xyz)
    x, y, z = xyz.T
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Iyz = -np.sum(masses * y * z)
    Ixz = -np.sum(masses * x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    return I

def get_principal_moi(I):
    Ip = np.linalg.eigvals(I)
    # Sort and convert principal moments of inertia to SI (kg.m2)
    Ip.sort()
    return Ip[::-1]

def get_self_euclidean_dist_matrix(x, y, z):
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    z = np.expand_dims(z, axis=1)
    dist = np.sqrt((x - x.T)**2 + (y - y.T)**2 + (z - z.T)**2)
    return dist

def get_principal_axis_lengths(xyz):
    xyz_centered = xyz - np.mean(xyz, axis=0, keepdims=True)
    cov = np.cov(xyz_centered, rowvar=False)
    eigval , eigvec = np.linalg.eig(cov)
    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:,idx]
    eigval = eigval[idx]
    pa = np.dot(xyz_centered, eigvec)
    pa_len = np.max(pa, axis=0) - np.min(pa, axis=0)
    return pa_len

def get_global_physical_props(mol, get_dist_matrix=True):
    # Get conformer information
    conf = mol.GetConformer()
    xyz = conf.GetPositions() / 10

    out = {}

    # Get distance matrix
    if get_dist_matrix:
        out["self_dist_matrix"] = get_self_euclidean_dist_matrix(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # Get the inertia from the mass
    C = Chem.Atom("C")
    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    masses_norm = (masses) / C.GetMass()
    I = get_inertia_matrix(masses_norm + EPS, xyz)
    out["Inertia_mass_a"], out["Inertia_mass_b"], out["Inertia_mass_c"] = np.real(get_principal_moi(I))

    # Get the inertia from the hydrogen distribution
    num_Hs = np.array([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
    IH = get_inertia_matrix(num_Hs + EPS, xyz)
    out["Inertia_valence_a"], out["Inertia_valence_b"], out["Inertia_valence_c"] = np.real(get_principal_moi(IH))

    # Get principal length
    out["length_a"], out["length_b"], out["length_c"] = np.real(get_principal_axis_lengths(xyz))

    # # electronegativity correlates too much with mass for the most common atoms
    # electronegativity = np.array([PERIODIC_TABLE["Electronegativity"][atom.GetAtomicNum()] for atom in mol.GetAtoms()])
    # electronegativity[np.isnan(electronegativity)] = 0
    # Ieneg = get_inertia_matrix(electronegativity + EPS, xyz)
    # out["Ieneg_a"], out["Ieneg_n"], out["Ieneg_c"] = np.real(get_principal_moi(Ieneg))

    # # Atomic radius correlates too much with mass for the most common atoms
    # rad = np.array([PERIODIC_TABLE["AtomicRadius"][atom.GetAtomicNum()] for atom in mol.GetAtoms()])
    # rad[np.isnan(rad)] = 0
    # Irad = get_inertia_matrix(rad + EPS, xyz)
    # out["Irad_a"], out["Irad_n"], out["Irad_c"] = np.real(get_principal_moi(Irad))


    # PBF: Plane of best fit
    out["Spherocity"] = CalcSpherocityIndex(mol)
    out["Plane_best_fit"] = CalcPBF(mol)

    return out

def generate_physics_dict_structure(mol, idx, skip_error=True):
    try:
        d = get_global_physical_props(mol, get_dist_matrix=False)
        d["idx"] = idx
    except Exception as e:
        if skip_error:
            print(f"Failed for molecule idx={idx} SMILES={MolToSmiles(mol)}")
            d = None
        else:
            raise ValueError(e)
    return d
