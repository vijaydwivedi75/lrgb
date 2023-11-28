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


### Update: Reassessment of LRGB

For a reassessment of the baselines on which LRGB were initially evaluated, we refer to [this paper](https://arxiv.org/abs/2309.00367) and thank @toenshoff for the PR on PCQM-Contact's evaluation metric.


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

Currently reported results (last update on Aug 10th, 2023)

<details>
  <summary> PascalVOC-SP (Node Classification) </summary>

  | Model | Test F1 (higher is better) | Reference | #params |
  | --- | --- | --- | --- |
  | Exphormer | 0.3975±0.0037 | [Shirzad, Velingker, Venkatachalam, et al, ICML 2023](https://openreview.net/forum?id=3Ge74dgjjU) | 509k |
  | GraphGPS | 0.3748±0.0109 | [Rampášek et al, NeurIPS 2022](https://openreview.net/forum?id=lMMaNf6oxKM) | 510k |
  | Cache-GNN+LapPE | 0.3462±0.0085 | [Ma et al, KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599260) | 500k | 
  | DRew-GatedGCN+LapPE | 0.3314±0.0024 | [Gutteridge et al, ICML 2023](https://openreview.net/forum?id=WEgjbJ6IDN) | 502k |
  | SAN+LapPE | 0.3230±0.0039 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 531k |
  | SAN+RWSE | 0.3216±0.0027 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 468k |
  | GatedGCN+LapPE+virtual node | 0.3103±0.0068 | [Cai et al, ICML 2023](https://openreview.net/forum?id=1EuHYKFPgA) | 502k |
  | GatedGCN | 0.2873±0.0219 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 502k |
  | GatedGCN+LapPE | 0.2860±0.0085 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 502k |
  | Transformer+LapPE | 0.2694±0.0098 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 501k |
  | GCNII | 0.1698±0.0080 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 492k |
  | GCN | 0.1268±0.0060 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 496k |
  | GINE | 0.1265±0.0076 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 505k |

</details>

<details>
  <summary> COCO-SP (Node Classification) </summary>

  | Model | Test F1 (higher is better) | Reference | #params |
  | --- | --- | --- | --- |
  | Exphormer | 0.3455±0.0009 | [Shirzad, Velingker, Venkatachalam, et al, ICML 2023](https://openreview.net/forum?id=3Ge74dgjjU) | 499k |
  | GraphGPS | 0.3412±0.0044 | [Rampášek et al, NeurIPS 2022](https://openreview.net/forum?id=lMMaNf6oxKM) | 516k |
  | Cache-GNN+LapPE | 0.2793±0.0033 | [Ma et al, KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599260) | 500k | 
  | GatedGCN | 0.2641±0.0045 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 509k |
  | Transformer+LapPE | 0.2618±0.0031 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 508k |
  | SAN+LapPE | 0.2592±0.0158 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 536k |
  | GatedGCN+LapPE | 0.2574±0.0034 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 509k |
  | SAN+RWSE | 0.2434±0.0156 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 474k |
  | GCNII | 0.1404±0.0011 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 505k |
  | GINE | 0.1339±0.0044 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 515k |
  | GCN | 0.0841±0.0010 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 509k |

</details>

<details>
  <summary> Peptides-func (Graph Classification) </summary>

  | Model | Test AP (higher is better) | Reference | #params |
  | --- | --- | --- | --- |
  | DRew-GCN+LapPE | 0.7150±0.0044 | [Gutteridge et al, ICML 2023](https://openreview.net/forum?id=WEgjbJ6IDN) | 502k |
  | GRIT | 0.6988±0.0082 | [Ma, Lin, et al, ICML 2023](https://openreview.net/forum?id=HjMdlNgybR) | 443k |
  | GraphMLP-Mixer | 0.6970±0.0080 | [He et al, ICML 2023](https://openreview.net/forum?id=l7yTbEWuOQ) | 397k |
  | Graph ViT | 0.6942±0.0075 | [He et al, ICML 2023](https://openreview.net/forum?id=l7yTbEWuOQ) | 692k |
  | MGT+WavePE | 0.6817±0.0064 | [Ngo, Hy, et al, 2023](https://arxiv.org/abs/2302.08647) | 499k |
  | PathNN | 0.6816±0.0026 | [Michel, Nikolentzos et al, ICML 2023](https://openreview.net/forum?id=5Purw053IP) | 510k |
  | GatedGCN+RWSE+virtual node | 0.6685±0.0062 | [Cai et al, ICML 2023](https://openreview.net/forum?id=1EuHYKFPgA) | 506k |
  | Cache-GNN+LapPE | 0.6671±0.0056 | [Ma et al, KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599260) | 500k | 
  | Graph Diffuser | 0.6651±0.0010 | [Glickman & Yahav, 2023](https://arxiv.org/abs/2303.00613) | 509k |
  | CIN++ | 0.6569±0.0117 | [Giusti et al, 2023](https://arxiv.org/abs/2306.03561) | ~500k |
  | GraphGPS | 0.6535±0.0041 | [Rampášek et al, NeurIPS 2022](https://openreview.net/forum?id=lMMaNf6oxKM) | 504k |
  | Exphormer | 0.6527±0.0043 | [Shirzad, Velingker, Venkatachalam, et al, ICML 2023](https://openreview.net/forum?id=3Ge74dgjjU) | 446k |
  | SAN+RWSE | 0.6439±0.0075 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 500k |
  | SAN+LapPE | 0.6384±0.0121 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 493k |
  | Transformer+LapPE | 0.6326±0.0126 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 488k |
  | GatedGCN+RWSE | 0.6069±0.0035 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 506k |
  | GCN | 0.5930±0.0023 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 508k |
  | GatedGCN | 0.5864±0.0077 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 509k |
  | GCNII | 0.5543±0.0078 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 505k |
  | GINE | 0.5498±0.0079 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 476k |

</details>

<details>
  <summary> Peptides-struct (Graph Regression) </summary>

  | Model | Test MAE (lower is better) | Reference | #params |
  | --- | --- | --- | --- |
  | Cache-GNN+LapPE | 0.2358±0.0013 | [Ma et al, KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599260) | 500k | 
  | Graph ViT | 0.2449±0.0016 | [He et al, ICML 2023](https://openreview.net/forum?id=l7yTbEWuOQ) | 561k |
  | MGT+WavePE | 0.2453±0.0025 | [Ngo, Hy, et al, 2023](https://arxiv.org/abs/2302.08647) | 499k |
  | GRIT | 0.2460±0.0012 | [Ma, Lin, et al, ICML 2023](https://openreview.net/forum?id=HjMdlNgybR) | 439k |
  | Graph Diffuser | 0.2461±0.0010 | [Glickman & Yahav, 2023](https://arxiv.org/abs/2303.00613) | 509k |
  | Exphormer | 0.2481±0.0007 | [Shirzad, Velingker, Venkatachalam, et al, ICML 2023](https://openreview.net/forum?id=3Ge74dgjjU) | 426k |
  | GCN+virtual node | 0.2488±0.0021 | [Cai et al, ICML 2023](https://openreview.net/forum?id=1EuHYKFPgA) | 508k |
  | Graph MLP-Mixer | 0.2494±0.0007 | [He et al, ICML 2023](https://openreview.net/forum?id=l7yTbEWuOQ) | 397k |
  | GraphGPS | 0.2500±0.0005 | [Rampášek et al, NeurIPS 2022](https://openreview.net/forum?id=lMMaNf6oxKM) | 504k |
  | CIN++ | 0.2523±0.0013 | [Giusti et al, 2023](https://arxiv.org/abs/2306.03561) | ~500k |
  | Transformer+LapPE | 0.2529±0.0016 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 488k |
  | DRew-GCN+LapPE | 0.2536±0.0015 | [Gutteridge et al, ICML 2023](https://openreview.net/forum?id=WEgjbJ6IDN) | 495k |
  | SAN+RWSE | 0.2545±0.0012 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 500k |
  | PathNN | 0.2545±0.0032 | [Michel, Nikolentzos et al, ICML 2023](https://openreview.net/forum?id=5Purw053IP) | 469k |
  | NPQ+GATv2 | 0.2589±0.0031 | [Jain et al, KLR Workshop at ICML, 2023](https://arxiv.org/abs/2307.09660) | NA |
  | SAN+LapPE | 0.2683±0.0043 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 493k |
  | GatedGCN+RWSE | 0.3357±0.0006 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 506k |
  | GatedGCN | 0.3420±0.0013 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 509k |
  | GCNII | 0.3471±0.0010 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 505k |
  | GCN | 0.3496±0.0013 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 508k |
  | GINE | 0.3547±0.0045 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 476k |

</details>

<details>
  <summary> PCQM-Contact (Link Prediction) </summary>

  | Model | Test MRR (higher is better) | Test Hits@1 | Test Hits@3 | Test Hits@10 | Reference | #params |
| --- | --- | --- | --- | --- | --- | --- |
| Exphormer | 0.3637±0.0020 |  |  |  | [Shirzad, Velingker, Venkatachalam, et al, ICML 2023](https://openreview.net/forum?id=3Ge74dgjjU) | 396k |
| Cache-GNN+RWSE | 0.3488±0.0008 | 0.1463±0.0011 | 0.4102±0.0008 | 0.8693±0.0008 | [Ma et al, KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599260) | 500k | 
| DRew-GCN | 0.3444±0.0017 |  |  |  | [Gutteridge et al, ICML 2023](https://openreview.net/forum?id=WEgjbJ6IDN) | 515k |
| Graph Diffuser | 0.3388±0.0011 | 0.1369±0.0012 | 0.4053±0.0011 | 0.8592±0.0007 | [Glickman & Yahav, 2023](https://arxiv.org/abs/2303.00613) | 521k |
| SAN+LapPE | 0.3350±0.0003 | 0.1355±0.0017 | 0.4004±0.0021 | 0.8478±0.0044 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 499k |
| SAN+RWSE | 0.3341±0.0006 | 0.1312±0.0016 | 0.4030±0.0008 | 0.8550±0.0024 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 509k |
| GraphGPS | 0.3337±0.0006 |  |  |  | [Rampášek et al, NeurIPS 2022](https://openreview.net/forum?id=lMMaNf6oxKM) | 513k |
| GatedGCN+RWSE | 0.3242±0.0008 | 0.1288±0.0013 | 0.3808±0.0006 | 0.8517±0.0005 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 524k |
| GCN | 0.3234±0.0006 | 0.1321±0.0007 | 0.3791±0.0004 | 0.8256±0.0006 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 504k |
| GatedGCN | 0.3218±0.0011 | 0.1279±0.0018 | 0.3783±0.0004 | 0.8433±0.0011 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 527k |
| GINE | 0.3180±0.0027 | 0.1337±0.0013 | 0.3642±0.0043 | 0.8147±0.0062 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 517k |
| Transformer+LapPE | 0.3174±0.0020 | 0.1221±0.0011 | 0.3679±0.0033 | 0.8517±0.0039 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 502k |
| GCNII | 0.3161±0.0004 | 0.1325±0.0009 | 0.3607±0.0003 | 0.8116±0.0009 | [Dwivedi et al, NeurIPS 2022](https://arxiv.org/abs/2206.08164) | 501k |

</details>


## Citation

If you find this work useful, please cite our paper:
```bibtex
@inproceedings{dwivedi2022LRGB,
  title={Long Range Graph Benchmark}, 
  author={Dwivedi, Vijay Prakash and Rampášek, Ladislav and Galkin, Mikhail and Parviz, Ali and Wolf, Guy and Luu, Anh Tuan and Beaini, Dominique},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=in7XC5RcjEn}
}
```
