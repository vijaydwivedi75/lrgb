#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    for SEED in {0..3}; do
        # script="sbatch ${slurm_directive} -J ${cfg_suffix}-${dataset} run/wrapper.sb ${main} --repeat 1 seed ${SEED} ${common_params}"
        script="sbatch ${slurm_directive} -J ${cfg_suffix}-${dataset} run/wrapper-narval.sb ${main} --repeat 1 seed ${SEED} ${common_params}"
        echo $script
        eval $script
    done
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done




################################################################################
##### Transformer+LapPE
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/GT"
slurm_directive="--time=0-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

# 1.1.1 Transformer+LapPE VOCSuperpixels slic 10
run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 Transformer+LapPE VOCSuperpixels slic 30
run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc name_tag Transformer-LapPE-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/GT"
slurm_directive="--time=0-23:59:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"

# 1.2.1 Transformer+LapPE COCOSuperpixels slic 10
run_repeats cocosuperpixels Transformer+LapPE "wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels Transformer+LapPE "wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels Transformer+LapPE "wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 Transformer+LapPE COCOSuperpixels slic 30
run_repeats cocosuperpixels Transformer+LapPE "wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels Transformer+LapPE "wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels Transformer+LapPE "wandb.project lrgb-coco name_tag Transformer-LapPE-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"



################################################################################
##### SAN
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/SAN"
slurm_directive="--time=2-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

# 1.1.1 SAN VOCSuperpixels slic 10
run_repeats vocsuperpixels SAN "wandb.project lrgb-voc name_tag SAN-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels SAN "wandb.project lrgb-voc name_tag SAN-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels SAN "wandb.project lrgb-voc name_tag SAN-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 SAN VOCSuperpixels slic 30
run_repeats vocsuperpixels SAN "wandb.project lrgb-voc name_tag SAN-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels SAN "wandb.project lrgb-voc name_tag SAN-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels SAN "wandb.project lrgb-voc name_tag SAN-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/SAN"
slurm_directive="--time=2-11:59:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"

# 1.2.1 SAN COCOSuperpixels slic 10
run_repeats cocosuperpixels SAN "wandb.project lrgb-coco name_tag SAN-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels SAN "wandb.project lrgb-coco name_tag SAN-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels SAN "wandb.project lrgb-coco name_tag SAN-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 SAN COCOSuperpixels slic 30
slurm_directive="--time=2-11:59:00 --mem=72G --gres=gpu:1 --cpus-per-task=4"  # SAN needs even more RAM on COCO-slic30
run_repeats cocosuperpixels SAN "wandb.project lrgb-coco name_tag SAN-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels SAN "wandb.project lrgb-coco name_tag SAN-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels SAN "wandb.project lrgb-coco name_tag SAN-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"



################################################################################
##### SAN-RWSE
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/SAN"
slurm_directive="--time=2-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

# 1.1.1 SAN VOCSuperpixels slic 10
run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc name_tag SAN-RWSE-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 SAN VOCSuperpixels slic 30
run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc name_tag SAN-RWSE-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc name_tag SAN-RWSE-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/SAN"
slurm_directive="--time=2-11:59:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"

# 1.2.1 SAN COCOSuperpixels slic 10
run_repeats cocosuperpixels SAN+RWSE "wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels SAN+RWSE "wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels SAN+RWSE "wandb.project lrgb-coco name_tag SAN-RWSE-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 SAN COCOSuperpixels slic 30
slurm_directive="--time=2-11:59:00 --mem=72G --gres=gpu:1 --cpus-per-task=4"  # SAN needs even more RAM on COCO-slic30
run_repeats cocosuperpixels SAN+RWSE "wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels SAN+RWSE "wandb.project lrgb-coco name_tag SAN-RWSE-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels SAN+RWSE "wandb.project lrgb-coco name_tag SAN-RWSE-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"
