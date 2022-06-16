# LRGB: Long Range Graph Benchmark

We present the **Long Range Graph Benchmark (LRGB)** with 5 graph learning datasets that arguably require
long-range reasoning to achieve strong performance in a given task. 
- PascalVOC-SP
- COCO-SP
- PCQM-Contact 
- Peptides-func
- Peptides-struct 

In this repo, we provide the source code to load the proposed datasets and run baseline experiments. 
The repo is based on [GraphGPS](https://github.com/rampasek/GraphGPS) which is built using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).


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