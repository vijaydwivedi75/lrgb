# This script will be run in the root directory.


### 1. For SAN, 4 seeds ######################

## 1.1 VOCSuperpixels

config=configs/SAN/vocsuperpixels-SAN.yaml

# 1.1.1 SAN VOCSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.1.2 SAN VOCSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait


## 1.2 COCOSuperpixels

config=configs/SAN/cocosuperpixels-SAN.yaml

# 1.2.1 SAN COCOSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.2.2 SAN COCOSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait