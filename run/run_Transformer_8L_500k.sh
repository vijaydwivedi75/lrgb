# This script will be run in the root directory.


### 1. For Transformer-LapPE, 4 seeds ######################

## 1.1 VOCSuperpixels

config=configs/GPS/vocsuperpixels-Transformer+LapPE.yaml

# 1.1.1 Transformer-LapPE VOCSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.1.2 Transformer-LapPE VOCSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait


## 1.2 COCOSuperpixels

config=configs/GPS/cocosuperpixels-Transformer+LapPE.yaml

# 1.2.1 Transformer-LapPE COCOSuperpixels slic 10

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10 &
done
wait

# 1.2.2 Transformer-LapPE COCOSuperpixels slic 30

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg $config device cuda:$SEED seed $SEED wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30 &
done
wait