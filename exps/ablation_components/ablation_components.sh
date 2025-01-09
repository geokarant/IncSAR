#!/bin/bash

JSON_FOLDER="./exps/ablation_components/aircraft"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done

JSON_FOLDER="./exps/ablation_components/mstar"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done
