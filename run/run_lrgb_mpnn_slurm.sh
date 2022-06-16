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
##### GCN
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/GCN"
slurm_directive="--time=0-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

# 1.1.1 GCN VOCSuperpixels slic 10
run_repeats vocsuperpixels GCN "wandb.project lrgb-voc name_tag GCN-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels GCN "wandb.project lrgb-voc name_tag GCN-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels GCN "wandb.project lrgb-voc name_tag GCN-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 GCN VOCSuperpixels slic 30
run_repeats vocsuperpixels GCN "wandb.project lrgb-voc name_tag GCN-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels GCN "wandb.project lrgb-voc name_tag GCN-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels GCN "wandb.project lrgb-voc name_tag GCN-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/GCN"
slurm_directive="--time=2-00:00:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"

# 1.2.1 GCN COCOSuperpixels slic 10
run_repeats cocosuperpixels GCN "wandb.project lrgb-coco name_tag GCN-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels GCN "wandb.project lrgb-coco name_tag GCN-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels GCN "wandb.project lrgb-coco name_tag GCN-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 GCN COCOSuperpixels slic 30
run_repeats cocosuperpixels GCN "wandb.project lrgb-coco name_tag GCN-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels GCN "wandb.project lrgb-coco name_tag GCN-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels GCN "wandb.project lrgb-coco name_tag GCN-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"



################################################################################
##### GINE
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/GINE"
slurm_directive="--time=0-15:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# 1.1.1 GINE VOCSuperpixels slic 10
run_repeats vocsuperpixels GINE "wandb.project lrgb-voc name_tag GINE-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels GINE "wandb.project lrgb-voc name_tag GINE-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels GINE "wandb.project lrgb-voc name_tag GINE-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 GINE VOCSuperpixels slic 30
run_repeats vocsuperpixels GINE "wandb.project lrgb-voc name_tag GINE-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels GINE "wandb.project lrgb-voc name_tag GINE-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels GINE "wandb.project lrgb-voc name_tag GINE-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/GINE"
slurm_directive="--time=2-10:00:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"
# 1.2.1 GINE COCOSuperpixels slic 10
run_repeats cocosuperpixels GINE "wandb.project lrgb-coco name_tag GINE-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels GINE "wandb.project lrgb-coco name_tag GINE-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels GINE "wandb.project lrgb-coco name_tag GINE-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 GINE COCOSuperpixels slic 30
run_repeats cocosuperpixels GINE "wandb.project lrgb-coco name_tag GINE-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels GINE "wandb.project lrgb-coco name_tag GINE-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels GINE "wandb.project lrgb-coco name_tag GINE-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"



################################################################################
##### GatedGCN
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/GatedGCN"
slurm_directive="--time=0-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

# 1.1.1 GatedGCN VOCSuperpixels slic 10
run_repeats vocsuperpixels GatedGCN "wandb.project lrgb-voc name_tag GatedGCN-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels GatedGCN "wandb.project lrgb-voc name_tag GatedGCN-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels GatedGCN "wandb.project lrgb-voc name_tag GatedGCN-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 GatedGCN VOCSuperpixels slic 30
run_repeats vocsuperpixels GatedGCN "wandb.project lrgb-voc name_tag GatedGCN-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels GatedGCN "wandb.project lrgb-voc name_tag GatedGCN-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels GatedGCN "wandb.project lrgb-voc name_tag GatedGCN-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/GatedGCN"
slurm_directive="--time=2-00:00:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"

# 1.2.1 GatedGCN COCOSuperpixels slic 10
run_repeats cocosuperpixels GatedGCN "wandb.project lrgb-coco name_tag GatedGCN-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels GatedGCN "wandb.project lrgb-coco name_tag GatedGCN-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels GatedGCN "wandb.project lrgb-coco name_tag GatedGCN-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 GatedGCN COCOSuperpixels slic 30
run_repeats cocosuperpixels GatedGCN "wandb.project lrgb-coco name_tag GatedGCN-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels GatedGCN "wandb.project lrgb-coco name_tag GatedGCN-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels GatedGCN "wandb.project lrgb-coco name_tag GatedGCN-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"



################################################################################
##### GatedGCN-LapPE
################################################################################

## 1.1 VOCSuperpixels

cfg_dir="configs/GatedGCN"
slurm_directive="--time=0-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

# 1.1.1 GatedGCN-LapPE VOCSuperpixels slic 10
run_repeats vocsuperpixels GatedGCN+LapPE "wandb.project lrgb-voc name_tag GatedGCN-LapPE-VOC-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats vocsuperpixels GatedGCN+LapPE "wandb.project lrgb-voc name_tag GatedGCN-LapPE-VOC-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats vocsuperpixels GatedGCN+LapPE "wandb.project lrgb-voc name_tag GatedGCN-LapPE-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.1.2 GatedGCN-LapPE VOCSuperpixels slic 30
run_repeats vocsuperpixels GatedGCN+LapPE "wandb.project lrgb-voc name_tag GatedGCN-LapPE-VOC-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats vocsuperpixels GatedGCN+LapPE "wandb.project lrgb-voc name_tag GatedGCN-LapPE-VOC-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats vocsuperpixels GatedGCN+LapPE "wandb.project lrgb-voc name_tag GatedGCN-LapPE-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


## 1.2 COCOSuperpixels

cfg_dir="configs/GatedGCN"
slurm_directive="--time=2-00:00:00 --mem=64G --gres=gpu:1 --cpus-per-task=4"

# 1.2.1 GatedGCN-LapPE COCOSuperpixels slic 10
run_repeats cocosuperpixels GatedGCN+LapPE "wandb.project lrgb-coco name_tag GatedGCN-LapPE-COCO-COOgraph-slic10 dataset.name edge_wt_only_coord dataset.slic_compactness 10"
run_repeats cocosuperpixels GatedGCN+LapPE "wandb.project lrgb-coco name_tag GatedGCN-LapPE-COCO-COOFEATgraph-slic10 dataset.name edge_wt_coord_feat dataset.slic_compactness 10"
run_repeats cocosuperpixels GatedGCN+LapPE "wandb.project lrgb-coco name_tag GatedGCN-LapPE-COCO-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2.2 GatedGCN-LapPE COCOSuperpixels slic 30
run_repeats cocosuperpixels GatedGCN+LapPE "wandb.project lrgb-coco name_tag GatedGCN-LapPE-COCO-COOgraph-slic30 dataset.name edge_wt_only_coord dataset.slic_compactness 30"
run_repeats cocosuperpixels GatedGCN+LapPE "wandb.project lrgb-coco name_tag GatedGCN-LapPE-COCO-COOFEATgraph-slic30 dataset.name edge_wt_coord_feat dataset.slic_compactness 30"
run_repeats cocosuperpixels GatedGCN+LapPE "wandb.project lrgb-coco name_tag GatedGCN-LapPE-COCO-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"
