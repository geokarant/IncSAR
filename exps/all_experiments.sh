#!/bin/bash

echo "Running basic_experiments.sh..."
bash ./exps/basic_experiments/basic_experiments.sh

echo "Running cross_dataset.sh..."
bash ./exps/cross_dataset/cross_dataset.sh

echo "Running ablation_backbones.sh..."
bash ./exps/exps_ablation_backbones/ablation_backbones.sh

echo "Running ablation_components.sh..."
bash ./exps/exps_ablation_incsar/ablation_components.sh

echo "Running ablation_portion.sh..."
bash ./exps/exps_ablation_portion/ablation_portion.sh

echo "All scripts have been executed.