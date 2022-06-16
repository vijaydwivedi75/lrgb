import numpy as np
import torch
import pickle
import time
import os
# %matplotlib inline
import matplotlib.pyplot as plt


import pickle

# %load_ext autoreload
# %autoreload 2

from superpixels import COCOSegDatasetDGL 

# from data.data import LoadData
from torch.utils.data import DataLoader
# from data.superpixels import VOCSegDataset




    
    
# print(len(dataset[1].train))
# print(len(dataset[1].val))
# print(len(dataset[1].test))

# print(dataset[1].train[0])
# print(dataset[1].val[0])
# print(dataset[1].test[0])

def dump_coco_pyg_source(dataset, graph_format, slic_compactness):
    vallist = []
    for data in dataset.val:
        # print(data)
        x = data[0].ndata['feat'] #x
        edge_attr = data[0].edata['feat'] #edge_attr
        edge_index = torch.stack(data[0].edges(), 0) #edge_index
        y = data[1] #y
        vallist.append((x, edge_attr, edge_index, y))

    trainlist = []
    for data in dataset.train:
        # print(data)
        x = data[0].ndata['feat'] #x
        edge_attr = data[0].edata['feat'] #edge_attr
        edge_index = torch.stack(data[0].edges(), 0) #edge_index
        y = data[1] #y
        trainlist.append((x, edge_attr, edge_index, y))

    testlist = []
    for data in dataset.test:
        # print(data)
        x = data[0].ndata['feat'] #x
        edge_attr = data[0].edata['feat'] #edge_attr
        edge_index = torch.stack(data[0].edges(), 0) #edge_index
        y = data[1] #y
        testlist.append((x, edge_attr, edge_index, y))
        
    print(len(trainlist), len(vallist), len(testlist))
    
    pyg_source_dir = './final_pkls_slic_cmpt_'+str(slic_compactness)+'/coco_superpixels_'+graph_format
    if not os.path.exists(pyg_source_dir):
        os.makedirs(pyg_source_dir)
    
    start = time.time()
    with open(pyg_source_dir+'/train.pickle','wb') as f:
        pickle.dump(trainlist,f)
    print('Time (sec):',time.time() - start) # 1.84s
    
    start = time.time()
    with open(pyg_source_dir+'/val.pickle','wb') as f:
        pickle.dump(vallist,f)
    print('Time (sec):',time.time() - start) # 0.29s
    
    start = time.time()
    with open(pyg_source_dir+'/test.pickle','wb') as f:
        pickle.dump(testlist,f)
    print('Time (sec):',time.time() - start) # 0.44s
    
    
    
DATASET_NAME = 'COCO'
graph_format = ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
    
# SLIC COMPACTNESS=30 ------------------------------------
dataset = []
for gf in graph_format:
    start = time.time()
    data = COCOSegDatasetDGL(DATASET_NAME, gf, 30) 
    print('Time (sec):',time.time() - start)
    dataset.append(data)
    
for idx, gf in enumerate(graph_format):
    dump_coco_pyg_source(dataset[idx], gf, 30)

    
# SLIC COMPACTNESS=10 ------------------------------------
dataset = []
for gf in graph_format:
    start = time.time()
    data = COCOSegDatasetDGL(DATASET_NAME, gf, 10) 
    print('Time (sec):',time.time() - start)
    dataset.append(data)
    
for idx, gf in enumerate(graph_format):
    dump_coco_pyg_source(dataset[idx], gf, 10)