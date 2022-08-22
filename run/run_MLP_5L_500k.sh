# This script will be run in the root directory.

### 1. MLP Baseline ######################

for SEED in {0..3}; do
    python main.py --cfg configs/MLP/peptides-func-MLP.yaml device cuda:$SEED seed $SEED wandb.project mlpbaseline-peptides name_tag MLP-peptides-func &
done
wait

for SEED in {0..3}; do
    python main.py --cfg configs/MLP/peptides-struct-MLP.yaml device cuda:$SEED seed $SEED wandb.project mlpbaseline-peptides name_tag MLP-peptides-struct &
done
wait

