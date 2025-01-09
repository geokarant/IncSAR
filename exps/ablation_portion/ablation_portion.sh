#!/bin/bash

JSON_FOLDER="./exps/ablation_portion/aircraft"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done

JSON_FOLDER="./exps/ablation_portion/mstar_b2_inc2"
for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done
