# This script will be run in the root directory.


### 1. For SAN-RWSE, 4 seeds ######################

## 1.1 VOCSuperpixels

config=configs/SAN/vocsuperpixels-SAN+RWSE.yaml

# 1.1.1 SAN-RWSE VOCSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-RWSE-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.1.2 SAN-RWSE VOCSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag SAN-RWSE-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait


## 1.2 COCOSuperpixels

config=configs/SAN/cocosuperpixels-SAN+RWSE.yaml

# 1.2.1 SAN-RWSE COCOSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-RWSE-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.2.2 SAN-RWSE COCOSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag SAN-RWSE-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait