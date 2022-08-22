# This script will be run in the root directory.

### 1. GCNII ######################

for SEED in {0..3}; do
    python main.py --cfg configs/GCNII/peptides-func-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-gcnii-8l name_tag GCNII-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/GCNII/peptides-struct-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-gcnii-8l name_tag GCNII-peptides-struct &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/GCNII/vocsuperpixels-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-gcnii-8l name_tag GCNII-PascalVOC-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/GCNII/cocosuperpixels-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-gcnii-8l name_tag GCNII-COCO-RBgraph-slic30 &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/GCNII/pcqm-contact-GCNII.yaml device cuda:$SEED seed $SEED wandb.project lrgb-gcnii-8l name_tag GCNII-PCQM-Contact &
done
wait

