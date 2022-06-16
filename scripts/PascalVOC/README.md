# Steps to prepare PascalVOC-SP dataset.

This directory contains the source codes for preparation of VOCSuperpixels from the original image dataset, along with respective visualizations.

## Steps to reproduce 

1. Run the complete notebook [./generate_vocsuperpixels_raw.ipynb](./generate_vocsuperpixels_raw.ipynb) to get the target pkl files.    
2. Run the complete notebook [./prepare_voc_pygsource.ipynb](./prepare_voc_pygsource.ipynb) that takes as input part of the files from Step 1 to get the target directories `./voc_superpixels_edge_wt_only_coord/`, `./voc_superpixels_edge_wt_coord_feat/`, and `./voc_superpixels_edge_wt_region_boundary/`. These 3 directories are zipped and stored on the server that hosts these files, and are eventually used as PyG source files for the PascalVOC dataset in [this file](../../graphgps/loader/dataset/voc_superpixels.py).  



## Visualization notebook
The notebook [./superpixels_visualization_VOC.ipynb](./superpixels_visualization_VOC.ipynb) can be used to visualize sample source VOC images with their superpixels based on different graph constructions methods employed.
