# This script will be run in the root directory.


### 1. For GatedGCN, 4 seeds ######################

## 1.1 VOCSuperpixels

config=configs/GatedGCN/vocsuperpixels-GatedGCN.yaml

# 1.1.1 GatedGCN VOCSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag GatedGCN-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag GatedGCN-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag GatedGCN-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.1.2 GatedGCN VOCSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag GatedGCN-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag GatedGCN-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag GatedGCN-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait


## 1.2 COCOSuperpixels

config=configs/GatedGCN/cocosuperpixels-GatedGCN.yaml

# 1.2.1 GatedGCN COCOSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag GatedGCN-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag GatedGCN-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag GatedGCN-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.2.2 GatedGCN COCOSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag GatedGCN-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag GatedGCN-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag GatedGCN-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait