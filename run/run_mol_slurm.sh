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

cfg_dir="configs/GCN"
## 1. Peptides
slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats peptides-func GCN "name_tag GCN-500k-peptides-func"
run_repeats peptides-struct GCN "name_tag GCN-500k-peptides-struct"

## 2. PCQM-Contact
slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats pcqm-contact GCN "name_tag GCN-500k-pcqmcontact"



################################################################################
##### GINE
################################################################################

cfg_dir="configs/GINE"
## 1. Peptides
slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats peptides-func GINE "name_tag GINE-500k-peptides-func"
run_repeats peptides-struct GINE "name_tag GINE-500k-peptides-struct"

## 2. PCQM-Contact
slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats pcqm-contact GINE "name_tag GINE-500k-pcqmcontact"



################################################################################
##### GatedGCN
################################################################################

cfg_dir="configs/GatedGCN"
## 1. Peptides
slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats peptides-func GatedGCN "name_tag GatedGCN-500k-peptides-func"
run_repeats peptides-func GatedGCN+RWSE "name_tag GatedGCN+RWSE-500k-peptides-func"
run_repeats peptides-struct GatedGCN "name_tag GatedGCN-500k-peptides-struct"
run_repeats peptides-struct GatedGCN+RWSE "name_tag GatedGCN+RWSE-500k-peptides-struct"

## 2. PCQM-Contact
slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats pcqm-contact GatedGCN "name_tag GatedGCN-500k-pcqmcontact"
run_repeats pcqm-contact GatedGCN+RWSE "name_tag GatedGCN-RWSE-500k-pcqmcontact"



################################################################################
##### Transformer+LapPE
################################################################################

cfg_dir="configs/GT"
## 1. Peptides
slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats peptides-func Transformer+LapPE "name_tag Transformer+LapPE-500k-peptides-func"
run_repeats peptides-struct Transformer+LapPE "name_tag Transformer+LapPE-500k-peptides-struct"

## 2. PCQM-Contact
slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats pcqm-contact Transformer+LapPE "name_tag Transformer+LapPE-500k-pcqmcontact"



################################################################################
##### SAN (LapPE and RWSE)
################################################################################

cfg_dir="configs/SAN"
## 1. Peptides
slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats peptides-func SAN "name_tag SAN-500k-peptides-func"
run_repeats peptides-struct SAN "name_tag SAN-500k-peptides-struct"
# SAN+RWSE
run_repeats peptides-func SAN+RWSE "name_tag SAN+RWSE-500k-peptides-func"
run_repeats peptides-struct SAN+RWSE "name_tag SAN+RWSE-500k-peptides-struct"


## 2. PCQM-Contact
slurm_directive="--time=2-12:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
run_repeats pcqm-contact SAN "name_tag SAN-500k-pcqmcontact"
run_repeats pcqm-contact SAN+RWSE "name_tag SAN+RWSE-500k-pcqmcontact"

