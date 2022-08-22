# LRGB: Long Range Graph Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2206.08164-b31b1b.svg)](https://arxiv.org/abs/2206.08164)

<img src="https://i.imgur.com/2LKoGbu.png" align="right" width="275"/>

We present the **Long Range Graph Benchmark (LRGB)** with 5 graph learning datasets that arguably require
long-range reasoning to achieve strong performance in a given task. 
- PascalVOC-SP
- COCO-SP
- PCQM-Contact 
- Peptides-func
- Peptides-struct 

In this repo, we provide the source code to load the proposed datasets and run baseline experiments. 
The repo is based on [GraphGPS](https://github.com/rampasek/GraphGPS) which is built using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).


### Overview of Datasets

|  Dataset | Domain  |  Task | Node Feat. (dim)  | Edge Feat. (dim) | Perf. Metric | 
|---|---|---|---|---|---|
| PascalVOC-SP| Computer Vision | Node Prediction | Pixel + Coord (14) | Edge Weight (1 or 2) | macro F1 |
| COCO-SP | Computer Vision | Node Prediction | Pixel + Coord (14) | Edge Weight (1 or 2) | macro F1 |
| PCQM-Contact | Quantum Chemistry | Link Prediction | Atom Encoder (9) | Bond Encoder (3) | Hits@K, MRR
| Peptides-func | Chemistry | Graph Classification | Atom Encoder (9) | Bond Encoder (3) | AP
| Peptides-struct | Chemistry | Graph Regression | Atom Encoder (9) | Bond Encoder (3) | MAE |


### Statistics of Datasets

|  Dataset | # Graphs  |  # Nodes | μ Nodes  | μ Deg. | # Edges | μ Edges | μ Short. Path | μ Diameter 
|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| PascalVOC-SP| 11,355 | 5,443,545 | 479.40 | 5.65 | 30,777,444 | 2,710.48 | 10.74±0.51 | 27.62±2.13 |
| COCO-SP | 123,286 | 58,793,216 | 476.88 | 5.65 | 332,091,902 | 2,693.67 | 10.66±0.55 | 27.39±2.14 |
| PCQM-Contact | 529,434 | 15,955,687 | 30.14 | 2.03 | 32,341,644 | 61.09 |4.63±0.63 | 9.86±1.79 |
| Peptides-func | 15,535 | 2,344,859 | 150.94 | 2.04 | 4,773,974 | 307.30 | 20.89±9.79 | 56.99±28.72 |
| Peptides-struct | 15,535 | 2,344,859 | 150.94 | 2.04 | 4,773,974 | 307.30 | 20.89±9.79 | 56.99±28.72 |


### Python environment setup with Conda

```bash
conda create -n lrgb python=3.9
conda activate lrgb

conda install pytorch=1.9 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.2 -c pyg -c conda-forge
conda install pandas scikit-learn

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

# Check https://www.dgl.ai/pages/start.html to install DGL based on your CUDA requirements
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html

pip install performer-pytorch
pip install torchmetrics==0.7.2
pip install ogb
pip install wandb

conda clean --all
```

### Running GraphGPS
```bash
conda activate lrgb

# Running GCN baseline for Peptides-func.
python main.py --cfg configs/GCN/peptides-func-GCN.yaml  wandb.use False

# Running SAN baseline for PascalVOC-SP.
python main.py --cfg configs/SAN/vocsuperpixels-SAN.yaml  wandb.use False

```

The scripts for all experiments are located in [run](./run) directory.

### W&B logging
To use W&B logging, set `wandb.use True` and have a `gtransformers` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`). 

<!-- ### Datasets Links

Following is a list of direct URL links of the proposed datasets that are used in the respective files in the [graphgps/loader/dataset](graphgps/loader/dataset) directory. Note that the links below are just for information and there is no need to manually download these source files. The files are automatically downloaded in the corresponding files in the [graphgps/loader/dataset](graphgps/loader/dataset) directory.

1. PascalVOC-SP: [link](https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=0)  
2. COCO-SP: [link](https://www.dropbox.com/s/r6ihg1f4pmyjjy0/coco_superpixels_edge_wt_region_boundary.zip?dl=0)  
3. PCQM-Contact: [link](https://datasets-public-research.s3.us-east-2.amazonaws.com/PCQM4M/pcqm4m-contact.tsv.gz)  
4. Peptides-func: [link](https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=0)  
5. Peptides-struct: [link](https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=0)   -->

### License Information

|  Dataset | Derived from  |  Original License | LRGB Release License  |
|---|---|---|---|
| PascalVOC-SP| Pascal VOC 2011 | Custom* | Custom* |
| COCO-SP | MS COCO | CC BY 4.0 | CC BY 4.0 |
| PCQM-Contact | PCQM4Mv2 | CC BY 4.0 | CC BY 4.0 |
| Peptides-func | SATPdb | CC BY-NC 4.0 | CC BY-NC 4.0 |
| Peptides-struct | SATPdb | CC BY-NC 4.0 | CC BY-NC 4.0 |


*[Custom License](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) for Pascal VOC 2011 (respecting Flickr terms of use)

### Leaderboards
The leaderboards of various models' performance on the datasets in LRGB are at [paperswithcode](https://paperswithcode.com/dataset/pascalvoc-sp).

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{dwivedi2022LRGB,
  title={Long Range Graph Benchmark}, 
  author={Dwivedi, Vijay Prakash and Rampášek, Ladislav and Galkin, Mikhail and Parviz, Ali and Wolf, Guy and Luu, Anh Tuan and Beaini, Dominique},
  journal={arXiv:2206.08164},
  year={2022}
}
```
