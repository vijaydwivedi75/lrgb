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
##### 16L, 1M params expts for VOCSuperpixels RBgraph-slic10 and RBgraph-slic30
################################################################################



################################################################################
##### Transformer+LapPE
################################################################################

## 1 VOCSuperpixels

cfg_dir="configs/GT"
slurm_directive="--time=1-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# 1.1 Transformer+LapPE VOCSuperpixels slic 10

run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc-16l gt.layers 16 name_tag Transformer-LapPE-16l-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 1.2 Transformer+LapPE VOCSuperpixels slic 30

run_repeats vocsuperpixels Transformer+LapPE "wandb.project lrgb-voc-16l gt.layers 16 name_tag Transformer-LapPE-16l-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


################################################################################
##### SAN
################################################################################

## 2 VOCSuperpixels

cfg_dir="configs/SAN"
slurm_directive="--time=2-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"  # WARNING: current cfg runs out of 40GB VRAM with L=16
# 2.1 SAN VOCSuperpixels slic 10

run_repeats vocsuperpixels SAN "wandb.project lrgb-voc-16l gt.layers 16 name_tag SAN-16l-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 2.2 SAN VOCSuperpixels slic 30

run_repeats vocsuperpixels SAN "wandb.project lrgb-voc-16l gt.layers 16 name_tag SAN-16l-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"


################################################################################
##### SAN-RWSE
################################################################################

## 3 VOCSuperpixels

cfg_dir="configs/SAN"
slurm_directive="--time=2-11:59:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"  # WARNING: current cfg runs out of 40GB VRAM with L=16

# 3.1 SAN VOCSuperpixels slic 10

run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc-16l gt.layers 16 name_tag SAN-RWSE-16l-VOC-RBgraph-slic10 dataset.name edge_wt_region_boundary dataset.slic_compactness 10"

# 3.2 SAN VOCSuperpixels slic 30

run_repeats vocsuperpixels SAN+RWSE "wandb.project lrgb-voc-16l gt.layers 16 name_tag SAN-RWSE-16l-VOC-RBgraph-slic30 dataset.name edge_wt_region_boundary dataset.slic_compactness 30"

