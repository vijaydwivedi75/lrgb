# This script will be run in the root directory.

### 1. GCN ######################

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCN/peptides-func-GCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCN-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCN/peptides-struct-GCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCN-peptides-struct &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCN/vocsuperpixels-GCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCN-PascalVOC-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCN/cocosuperpixels-GCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCN-COCO-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCN/pcqm-contact-GCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCN-PCQM-Contact &
done
wait


### 2. GCNII ######################

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCNII/peptides-func-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCNII-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCNII/peptides-struct-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCNII-peptides-struct &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCNII/vocsuperpixels-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCNII-PascalVOC-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCNII/cocosuperpixels-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCNII-COCO-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GCNII/pcqm-contact-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GCNII-PCQM-Contact &
done
wait

### 3. GatedGCN ######################

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/peptides-func-GatedGCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/peptides-func-GatedGCN+RWSE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-RWSE-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/peptides-struct-GatedGCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-peptides-struct &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/peptides-struct-GatedGCN+RWSE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-RWSE-peptides-struct &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/vocsuperpixels-GatedGCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-PascalVOC-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/vocsuperpixels-GatedGCN+LapPE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-LapPE-PascalVOC-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/cocosuperpixels-GatedGCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-COCO-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/cocosuperpixels-GatedGCN+LapPE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-LapPE-COCO-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/pcqm-contact-GatedGCN.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-PCQM-Contact &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GatedGCN/pcqm-contact-GatedGCN+RWSE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GatedGCN-RWSE-PCQM-Contact &
done
wait

### 4. GINE ######################

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GINE/peptides-func-GINE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GINE-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GINE/peptides-struct-GINE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GINE-peptides-struct &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GINE/vocsuperpixels-GINE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GINE-PascalVOC-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GINE/cocosuperpixels-GINE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GINE-COCO-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/expts_l2/GINE/pcqm-contact-GINE.yaml device cuda:$SEED seed $SEED wandb.project lrgb-2l name_tag GINE-PCQM-Contact &
done
wait