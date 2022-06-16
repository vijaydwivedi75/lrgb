## Steps to prepare COCO-SP dataset. 

- Install dgl and scikit-image through `conda install -c dglteam dgl-cuda10.2` and `conda install scikit-image`.  
- Install cython through `python -m pip install Cython`.  (needed for the cocoapi `make`).  
- Download the repo https://github.com/cocodataset/cocoapi in the current directory.    
- Then run `make` inside the `cocoapi/PythonAPI` directory.    
- Run the python file [generate_cocosuperpixels_raw.py](generate_cocosuperpixels_raw.py) to generate the source pkl files.     
- Run the python file [prepare_coco_pygsource.py](prepare_coco_pygsource.py) to generate the final source files for the PyG dataset class.    

