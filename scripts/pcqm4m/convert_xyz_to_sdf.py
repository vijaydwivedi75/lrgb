from openbabel import openbabel
import os
from black import main
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import fsspec
import pandas as pd
from ogb.utils.url import decide_download, download_url, extract_zip
import torch
from shutil import make_archive, rmtree

def main():
    XYZ_URL = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip"
    XYZ_LOCAL = "scripts/pcqm4m/pcqm4m-v2_xyz.zip"
    # RAW_URL = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
    # RAW_URL = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip"
    ROOT = "."

    # Read the raw data, and get the id, smiles, homo-lumo gap
    path = download_url(XYZ_URL, ROOT)
    extract_zip(path, ROOT)
    os.unlink(path)

    # Read the xyz files and get the id, positions, atoms, and pair-wise distances
    xyz_files = get_files_from_compressed_dir(XYZ_LOCAL, ext=".xyz")
    convert_subfolders_mol_type(xyz_files, in_type="xyz", out_type="sdf", delete_unzipped_folder=True)

    pass

def get_files_from_compressed_dir(dir, ext):
    with fsspec.open(dir) as f:
        with BytesIO(f.read()) as b, ZipFile(b) as zip_folder:
            list_of_files = zip_folder.namelist()
        files = [file for file in list_of_files if file.endswith(ext)]
    return files

def get_files_from_dir(dir, ext):
    files = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files

def convert_subfolders_mol_type(xyz_files, in_type, out_type, delete_unzipped_folder):
    unique_dirs, unique_inv = np.unique([os.path.dirname(file) for file in xyz_files], return_inverse=True)
    for ii, dir in enumerate(unique_dirs):
        new_dir = os.path.join(out_type, dir)
        dir_idx = np.where(unique_inv == ii)[0]
        these_files = [xyz_files[jj] for jj in dir_idx]
        out = Parallel(n_jobs=-1)(delayed(convert_molecular_files)(file, new_dir, in_type, out_type) for file in tqdm(these_files))
        if sum(out) > 0:
            make_archive(base_name=new_dir, format="zip", root_dir=os.path.dirname(new_dir), base_dir=dir)
            if delete_unzipped_folder:
                rmtree(new_dir)

def convert_molecular_files(filename, new_dir, in_type, out_type):
    out_file = os.path.join(new_dir, os.path.basename(filename)[:-4] + f".{out_type}")
    os.makedirs(new_dir, exist_ok=True)
    conv=openbabel.OBConversion()
    conv.SetInAndOutFormats(in_type, out_type)
    conv.OpenInAndOutFiles(filename, out_file)
    return conv.Convert()


if __name__ == "__main__":

    main()
    print("Done!")

