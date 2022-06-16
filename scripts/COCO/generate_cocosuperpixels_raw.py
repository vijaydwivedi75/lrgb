import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

import random
import scipy
import pickle
from skimage.segmentation import slic
from skimage.future import graph
from skimage import filters, color

import scipy.ndimage
import scipy.spatial
from scipy.spatial.distance import cdist

import time
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from joblib import delayed


from cocoapi.PythonAPI.pycocotools.coco import COCO
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)



from tqdm.auto import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    """A helper class for adding tqdm progressbar to the joblib library."""
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        
        
        
def process_single_image_slic(params):
    
    #img, index, n_images, args, to_print, shuffle = params
    img, mask, args, shuffle = params
    args_split, args_seed, args_n_sp, args_compactness = args
    img_original = img
    mask_original = mask

    random.seed(args_seed)
    np.random.seed(args_seed)
    
    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    n_sp_extracted = args_n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    
    # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    n_sp_query = args_n_sp + 50
    
    while n_sp_extracted > args_n_sp:
        superpixels = slic(img, n_segments=n_sp_query, compactness=args_compactness, multichannel=len(img.shape) > 2, start_label=0)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels

    assert n_sp_extracted <= args_n_sp and n_sp_extracted > 0, (args_split, n_sp_extracted, args_n_sp)
    
    # make sure superpixel indices are numbers from 0 to n-1
    assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))  

    # Creating region adjacency graph based on boundary
    gimg = color.rgb2gray(img_original)
    edges = filters.sobel(gimg)
    
    try:
        g = graph.rag_boundary(superpixels, edges)
    except ValueError: # Error thrown when graph size is perhaps 1
        print("ignored graph")
        g = nx.complete_graph(sp_indices) # so ignoring these for now and placing dummy info
        nx.set_edge_attributes(g, 0., "weight")
        nx.set_edge_attributes(g, 0, "count")
    
    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        std_value = np.zeros(n_ch)
        max_value = np.zeros(n_ch)
        min_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
            std_value[c] = np.std(img[:, :, c][mask])
            max_value[c] = np.max(img[:, :, c][mask])
            min_value[c] = np.min(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        
        sp_intensity.append(np.concatenate((avg_value,
                                           std_value,
                                           max_value,
                                           min_value), -1))
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)

    rag_boundary_graphs = dgl.from_networkx(g.to_directed(),edge_attrs=['weight', 'count'])
    sp_data = sp_intensity, sp_coord, sp_order
    
    """
    # NODE LABELING
    : using the coord value of the superpixel node to select the 
      corresponding label from the ground truth pixel (segmentation mask)
    """
    # for i, img in enumerate(data_set.mask_list):
    # coord = sp_data[1]                            # the x and y coord of the superpixel node (float)
    sp_x_coord = np.rint(sp_data[1][:,0]).astype(np.int16)    # the rounded x coord of the superpixel node (int)
    sp_y_coord = np.rint(sp_data[1][:,1]).astype(np.int16)    # the rounded y coord of the superpixel node (int)

    # labeling the superpixel node with the same value of the original pixel 
    # ground truth  that is on the mean coord of the superpixel node
    sp_node_labels = np.array(
        [mask_original[sp_x_coord[_]][sp_y_coord[_]] for _ in range(len(sp_x_coord))], dtype=np.uint8)

    return rag_boundary_graphs, sp_data, sp_node_labels




"""
    COCO categories: 
    person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop
    sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack
    umbrella handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball
    glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana
    apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed
    dining table toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink
    refrigerator book clock vase scissors teddy bear hair drier toothbrush
"""

class COCO_Images_Masks(data.Dataset):
    def __init__(self, mode, compactness, root='./cocoapi'):
        self.root = root
        self.mode = mode
        # self.all_superpixels = []
        self.all_rag_boundary_graphs = []
        self.all_sp_data = []
        self.all_sp_node_labels = []
        
        self.n_sp = 500
        self.compactness = compactness
        self.seed = 41
        self.out_dir = '.'
        self.dataset = 'COCO'
        
        self.args = self.mode, self.seed, self.n_sp, self.compactness
        
        self.num_images = self._pack_images_masks(mode)
        
    def _pack_images_masks(self, mode):
        # in this paper, we train on the train set and evaluate on the val set
        assert mode in ['train', 'val']
        
        dataType = 'val2017' if mode == 'val' else 'train2017'
        annFile = '{}/annotations/instances_{}.json'.format(self.root, dataType)
        
        # initialize COCO api for instance annotations
        coco=COCO(annFile)
        
        imgIds = coco.getImgIds()
        cat_ids = coco.getCatIds()
        
        all_imgs = coco.loadImgs(imgIds)
        
        parallel = ProgressParallel(n_jobs=8, batch_size=32, prefer='threads', use_tqdm=True, total=len(all_imgs))
        all_imgs_outs = parallel(delayed(self.get_superpixel_graph)(imginfo,
                                                                    dataType=dataType,
                                                                    coco=coco,
                                                                    cat_ids=cat_ids) for imginfo in all_imgs)
        
        for out in all_imgs_outs:
            # self.all_superpixels.append(out[0])
            self.all_rag_boundary_graphs.append(out[0])
            self.all_sp_data.append(out[1])
            self.all_sp_node_labels.append(out[2])
        
        with open('%s/%s_%dsp_%dcmpt_%s.pkl' % (self.out_dir, self.dataset, self.n_sp, self.compactness, self.mode), 'wb') as f:
            pickle.dump((self.all_sp_node_labels, self.all_sp_data), f, protocol=4)
        with open('%s/%s_%dsp_%dcmpt_%s_rag_boundary_graphs.pkl' % (self.out_dir, self.dataset, self.n_sp, self.compactness, self.mode), 'wb') as f:
            pickle.dump(self.all_rag_boundary_graphs, f, protocol=4)
            
        # with open('%s/sample/%s_%dsp_%s.pkl' % (self.out_dir, self.dataset, self.n_sp, self.mode), 'wb') as f:
        #     pickle.dump((self.all_sp_node_labels, self.all_sp_data), f, protocol=2)
        # with open('%s/sample/%s_%dsp_%s_superpixels.pkl' % (self.out_dir, self.dataset, self.n_sp, self.mode), 'wb') as f:
        #     pickle.dump(self.all_superpixels, f, protocol=2)
        # with open('%s/sample/%s_%dsp_%s_rag_boundary_graphs.pkl' % (self.out_dir, self.dataset, self.n_sp, self.mode), 'wb') as f:
        #     pickle.dump(self.all_rag_boundary_graphs, f, protocol=2)    
        
        count_data_len = len(all_imgs_outs)
        del self.all_sp_data, self.all_sp_node_labels
        
        return count_data_len

    def get_superpixel_graph(self, img_meta_info, dataType, coco, cat_ids):
        # img = io.imread(img_meta_info['coco_url']) # This command actually fetches the img from url each time
        img = Image.open(os.path.join(self.root, 'images', dataType, img_meta_info['file_name'])).convert('RGB')

        anns_ids = coco.getAnnIds(imgIds=img_meta_info['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = np.zeros((img_meta_info['height'],img_meta_info['width']))
        for ann in anns:
            mask = np.maximum(mask,coco.annToMask(ann)*ann['category_id'])
            
        return process_single_image_slic((np.array(img, np.uint8),
                                          np.array(mask, np.uint8),
                                          self.args,
                                          False))
    
    def __getitem__(self, index):
        raise NotImplementedError
        # return self.img_list[index], self.mask_list[index]

    def __len__(self):
        return self.num_images
    
    
    
    
    
t0 = time.time()
print("[I] Reading and loading COCO 2017 Images and Masks for VAL set for sp=500, cmpt=10..")
COCO_Images_Masks('val', compactness=10)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))

t0 = time.time()
print("[I] Reading and loading COCO 2017 Images and Masks for TRAIN set for sp=500, cmpt=10..")
COCO_Images_Masks('train', compactness=10)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))

t0 = time.time()
print("[I] Reading and loading COCO 2017 Images and Masks for VAL set for sp=500, cmpt=30..")
COCO_Images_Masks('val', compactness=30)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))

t0 = time.time()
print("[I] Reading and loading COCO 2017 Images and Masks for TRAIN set for sp=500, cmpt=30..")
COCO_Images_Masks('train', compactness=30)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))